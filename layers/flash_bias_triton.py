"""
FlashBias: Fast Computation of Attention with Bias
Paper: https://arxiv.org/abs/2505.12044

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.

Acknowledgments:
- @triDao: the code is largely based on his implementation https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
"""

import math
import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [32, 64, 128]
    for BN in [32, 64, 128]
    for s in ([1])
    for w in [4]  # sometimes warp=8 may casue race condition, damage correctness
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    if BLOCK_M != BLOCK_N:
        return False
    return True


@triton.autotune(
    list(filter(keep, configs)),
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "MASK_TYPE", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM",
         "BLOCK_RANKDIM"]
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_RANKDIM": lambda args: args["rankdim"] == args["BLOCK_RANKDIM"],
    }
)
@triton.jit
def _fwd_kernel(
        Q,
        K,
        V,
        Q_Bias,
        K_Bias,
        Mask,
        Out,
        Lse,
        TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_q_bb,
        stride_q_bh,
        stride_q_bm,
        stride_k_bb,
        stride_k_bh,
        stride_k_bn,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_ob,
        stride_oh,
        stride_om,
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        rankdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        MASK_TYPE: tl.constexpr,
        BIAS_TYPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        BLOCK_RANKDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_RANKDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_RANKDIM)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
            Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
            K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
            V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE is not None:
        q_bias_ptrs = (
                Q_Bias + off_b * stride_q_bb + off_h * stride_q_bh + (offs_m[:, None] * stride_q_bm + offs_r[None, :])
        )
        k_bias_ptrs = (
                K_Bias + off_b * stride_k_bb + off_h * stride_k_bh + (offs_n[:, None] * stride_k_bn + offs_r[None, :])
        )
    if MASK_TYPE == "vector":
        m_ptrs = Mask + off_b * stride_mb + off_h * stride_mh + offs_n
    elif MASK_TYPE == "matrix":
        m_ptrs = (
                Mask
                + off_b * stride_mb
                + off_h * stride_mh
                + (offs_m[:, None] * stride_mm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        if BIAS_TYPE is not None:
            if EVEN_RANKDIM:
                q_bias = tl.load(q_bias_ptrs).to(tl.float32)
            else:
                q_bias = tl.load(q_bias_ptrs, mask=offs_r[None, :] < rankdim, other=0.0).to(tl.float32)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
        if BIAS_TYPE is not None:
            if EVEN_RANKDIM:
                q_bias = tl.load(q_bias_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0).to(tl.float32)
            else:
                q_bias = tl.load(
                    q_bias_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_r[None, :] < rankdim), other=0.0
                ).to(tl.float32)
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
            if BIAS_TYPE is not None:
                if EVEN_RANKDIM:
                    k_bias = tl.load(k_bias_ptrs + start_n * stride_k_bn).to(tl.float32)
                else:
                    k_bias = tl.load(k_bias_ptrs + start_n * stride_k_bn, mask=offs_r[None, :] < rankdim, other=0.0).to(
                        tl.float32)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
            if BIAS_TYPE is not None:
                if EVEN_RANKDIM:
                    k_bias = tl.load(
                        k_bias_ptrs + start_n * stride_k_bn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    k_bias = tl.load(
                        k_bias_ptrs + start_n * stride_k_bn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_r[None, :] < rankdim),
                        other=0.0,
                    ).to(tl.float32)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        assert q.shape[1] == k.shape[1], "Inner dimension must match for matrix multiplication"
        k_t = tl.trans(k)
        qk += tl.dot(q, k_t)
        # multiply the softmax scale 1 or 1/sqrt(d) here before adding bias or mask
        qk = qk * softmax_scale

        # processing bias term
        if BIAS_TYPE is not None:
            assert q_bias.shape[1] == k_bias.shape[1], "Inner dimension must match for matrix multiplication"
            k_bias_t = tl.trans(k_bias)
            # tl.device_print("Tensor Q_bias: ", q_bias)
            # tl.device_print("Tensor K_bias: ", k_bias)
            qk += tl.dot(q_bias, k_bias_t)

        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if MASK_TYPE != "none":
            if MASK_TYPE == "vector":
                if EVEN_N:
                    mask = tl.load(m_ptrs + start_n).to(tl.float32)
                else:
                    mask = tl.load(
                        m_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                mask = mask[None, :]
            elif MASK_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    mask = tl.load(m_ptrs + start_n).to(tl.float32)
                else:
                    mask = tl.load(
                        m_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                             & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk + mask
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
            Out
            + off_b * stride_ob
            + off_h * stride_oh
            + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
        Out,
        DO,
        Delta,
        stride_ob,
        stride_oh,
        stride_om,
        stride_dob,
        stride_doh,
        stride_dom,
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        headdim,
        BLOCK_M: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv_dk_bias(
        dk_ptrs,
        dk_bias_ptrs,
        dv_ptrs,
        dk,
        dk_bias,
        dv,
        offs_n,
        offs_d,
        offs_r,
        seqlen_k,
        headdim,
        rankdim,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_RANKDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        if dk_bias_ptrs is not None:
            if EVEN_RANKDIM:
                tl.store(dk_bias_ptrs, dk_bias)
            else:
                tl.store(dk_bias_ptrs, dk_bias, mask=offs_r[None, :] < rankdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
        if dk_bias_ptrs is not None:
            if EVEN_RANKDIM:
                tl.store(dk_bias_ptrs, dk_bias, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.store(dk_bias_ptrs, dk_bias, mask=(offs_n[:, None] < seqlen_k) & (offs_r[None, :] < rankdim))


@triton.jit
def _bwd_kernel_one_col_block(
        start_n,
        Q,
        K,
        Q_Bias,
        K_Bias,
        V,
        Mask,
        DO,
        DQ,
        DK,
        DQ_Bias,
        DK_Bias,
        DV,
        LSE,
        D,
        softmax_scale,
        stride_qm,
        stride_kn,
        stride_q_bm,
        stride_k_bn,
        stride_vn,
        stride_mm,
        stride_dom,
        stride_dqm,
        stride_dkn,
        stride_dq_bm,
        stride_dk_bn,
        stride_dvn,
        seqlen_q,
        seqlen_k,
        headdim,
        rankdim,
        ATOMIC_ADD: tl.constexpr,
        MASK_TYPE: tl.constexpr,
        BIAS_TYPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        BLOCK_RANKDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_RANKDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_r = tl.arange(0, BLOCK_RANKDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    if BIAS_TYPE is not None:
        q_bias_ptrs = Q_Bias + (offs_qm[:, None] * stride_q_bm + offs_r[None, :])
        k_bias_ptrs = K_Bias + (offs_n[:, None] * stride_k_bn + offs_r[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE is not None:
        dq_bias_ptrs = DQ_Bias + (offs_qm[:, None] * stride_dq_bm + offs_r[None, :])
    if MASK_TYPE == "vector":
        m_ptrs = Mask + offs_n
    elif MASK_TYPE == "matrix":
        m_ptrs = Mask + (offs_qm[:, None] * stride_mm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if BIAS_TYPE is not None:
        dk_bias = tl.zeros([BLOCK_N, BLOCK_RANKDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        if BIAS_TYPE is not None:
            dk_bias_ptrs = DK_Bias + (offs_n[:, None] * stride_dk_bn + offs_r[None, :])
            _bwd_store_dk_dv_dk_bias(
                dk_ptrs,
                dk_bias_ptrs,
                dv_ptrs,
                dk,
                dk_bias,
                dv,
                offs_n,
                offs_d,
                offs_r,
                seqlen_k,
                headdim,
                rankdim,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                EVEN_RANKDIM=EVEN_RANKDIM,
            )
        else:
            _bwd_store_dk_dv_dk_bias(
                dk_ptrs,
                None,
                dv_ptrs,
                dk,
                None,
                dv,
                offs_n,
                offs_d,
                offs_r,
                seqlen_k,
                headdim,
                rankdim,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                EVEN_RANKDIM=EVEN_RANKDIM,
            )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        if BIAS_TYPE is not None:
            if EVEN_RANKDIM:
                k_bias = tl.load(k_bias_ptrs)
            else:
                k_bias = tl.load(k_bias_ptrs, mask=offs_r[None, :] < rankdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
        if BIAS_TYPE is not None:
            if EVEN_RANKDIM:
                k_bias = tl.load(k_bias_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            else:
                k_bias = tl.load(
                    k_bias_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_r[None, :] < rankdim), other=0.0
                )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        if BIAS_TYPE is not None:
            if EVEN_M & EVEN_RANKDIM:
                q_bias = tl.load(q_bias_ptrs)
            else:
                if EVEN_RANKDIM:
                    q_bias = tl.load(q_bias_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                else:
                    q_bias = tl.load(
                        q_bias_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_r[None, :] < rankdim),
                        other=0.0,
                    )
        # recompute p = softmax(qk, dim=-1).T
        # qk = tl.dot(q, k, trans_b=True)
        k_t = tl.trans(k)
        qk = tl.dot(q, k_t) * softmax_scale
        if BIAS_TYPE is not None:
            k_bias_t = tl.trans(k_bias)
            qk += tl.dot(q_bias, k_bias_t)
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if MASK_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if MASK_TYPE == "vector":
                if EVEN_N:
                    mask = tl.load(m_ptrs).to(tl.float32)
                else:
                    mask = tl.load(m_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                mask = mask[None, :]
            elif MASK_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    mask = tl.load(m_ptrs).to(tl.float32)
                else:
                    mask = tl.load(
                        m_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk + mask
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        p_t = tl.trans(p.to(do.dtype))
        dv += tl.dot(p_t, do)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        v_t = tl.trans(v)
        dp = tl.dot(do, v_t)
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None])).to(q.dtype)
        # compute dk = dot(ds.T, q)
        ds_t = tl.trans(ds)
        dk += tl.dot(ds_t, q) * softmax_scale
        if BIAS_TYPE is not None:
            dk_bias += tl.dot(ds_t, q_bias)
        # compute dq
        if not (
                EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k) * softmax_scale
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
                if BIAS_TYPE is not None:
                    dq_bias = tl.load(dq_bias_ptrs, eviction_policy="evict_last")
                    dq_bias += tl.dot(ds, k_bias)
                    tl.store(dq_bias_ptrs, dq_bias, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k) * softmax_scale
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                    if BIAS_TYPE is not None:
                        dq_bias = tl.load(
                            dq_bias_ptrs,
                            mask=offs_m_curr[:, None] < seqlen_q,
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        dq_bias += tl.dot(ds, k_bias)
                        tl.store(
                            dq_bias_ptrs,
                            dq_bias,
                            mask=offs_m_curr[:, None] < seqlen_q,
                            eviction_policy="evict_last",
                        )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k) * softmax_scale
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
                    if BIAS_TYPE is not None:
                        dq_bias = tl.load(
                            dq_bias_ptrs,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_r[None, :] < rankdim),
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        dq_bias += tl.dot(ds, k_bias)
                        tl.store(
                            dq_bias_ptrs,
                            dq_bias,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_r[None, :] < rankdim),
                            eviction_policy="evict_last",
                        )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k) * softmax_scale
            if BIAS_TYPE is not None:
                dq_bias = tl.dot(ds, k_bias)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
            if BIAS_TYPE is not None:
                if EVEN_M & EVEN_RANKDIM:
                    tl.atomic_add(dq_bias_ptrs, dq_bias)
                else:
                    if EVEN_RANKDIM:
                        tl.atomic_add(dq_bias_ptrs, dq_bias, mask=offs_m_curr[:, None] < seqlen_q)
                    else:
                        tl.atomic_add(
                            dq_bias_ptrs,
                            dq_bias,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_r[None, :] < rankdim),
                        )
            # tl.device_print("Tensor DQ_bias: ", dq_bias)
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        if BIAS_TYPE is not None:
            dq_bias_ptrs += BLOCK_M * stride_dq_bm
            q_bias_ptrs += BLOCK_M * stride_q_bm
        do_ptrs += BLOCK_M * stride_dom
        if MASK_TYPE == "matrix":
            m_ptrs += BLOCK_M * stride_mm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    if BIAS_TYPE is not None:
        dk_bias_ptrs = DK_Bias + (offs_n[:, None] * stride_dk_bn + offs_r[None, :])
        _bwd_store_dk_dv_dk_bias(
            dk_ptrs,
            dk_bias_ptrs,
            dv_ptrs,
            dk,
            dk_bias,
            dv,
            offs_n,
            offs_d,
            offs_r,
            seqlen_k,
            headdim,
            rankdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_RANKDIM=EVEN_RANKDIM,
        )
    else:
        _bwd_store_dk_dv_dk_bias(
            dk_ptrs,
            None,
            dv_ptrs,
            dk,
            None,
            dv,
            offs_n,
            offs_d,
            offs_r,
            seqlen_k,
            headdim,
            rankdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_RANKDIM=EVEN_RANKDIM,
        )


def init_to_zero(*names):
    def hook(nargs):
        for name in names:
            buffer = nargs.get(name)
            if buffer is not None:
                buffer.zero_()

    return hook


@triton.autotune(
    configs=[
        *[triton.Config(
            {"BLOCK_M": m, "BLOCK_N": n, "SEQUENCE_PARALLEL": seq_parallel},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ", "DQ_Bias"),
        )
            for m, n in [(32, 32), (64, 64), (128, 128)]
            for seq_parallel in [True, False]
        ],
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "MASK_TYPE", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM",
         "BLOCK_RANKDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_RANKDIM": lambda args: args["rankdim"] == args["BLOCK_RANKDIM"],
    }
)
@triton.jit
def _bwd_kernel(
        Q,
        K,
        V,
        Q_Bias,
        K_Bias,
        Mask,
        DO,
        DQ,
        DQ_Bias,
        DK,
        DK_Bias,
        DV,
        LSE,
        D,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_q_bb,
        stride_q_bh,
        stride_q_bm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_k_bb,
        stride_k_bh,
        stride_k_bn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dq_bb,
        stride_dq_bh,
        stride_dq_bm,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dk_bb,
        stride_dk_bh,
        stride_dk_bn,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        rankdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        MASK_TYPE: tl.constexpr,
        BIAS_TYPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        BLOCK_RANKDIM: tl.constexpr,
        SEQUENCE_PARALLEL: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_RANKDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    if BIAS_TYPE is not None:
        Q_Bias += off_b * stride_q_bb + off_h * stride_q_bh
        K_Bias += off_b * stride_k_bb + off_h * stride_k_bh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    if BIAS_TYPE is not None:
        DQ_Bias += off_b * stride_dq_bb + off_h * stride_dq_bh
        DK_Bias += off_b * stride_dk_bb + off_h * stride_dk_bh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if MASK_TYPE != "none":
        Mask += off_b * stride_mb + off_h * stride_mh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                Q_Bias,
                K_Bias,
                V,
                Mask,
                DO,
                DQ,
                DK,
                DQ_Bias,
                DK_Bias,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_q_bm,
                stride_k_bn,
                stride_vn,
                stride_mm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dq_bm,
                stride_dk_bn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                rankdim,
                ATOMIC_ADD=False,
                MASK_TYPE=MASK_TYPE,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                BLOCK_RANKDIM=BLOCK_RANKDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                EVEN_RANKDIM=EVEN_RANKDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            Q_Bias,
            K_Bias,
            V,
            Mask,
            DO,
            DQ,
            DK,
            DQ_Bias,
            DK_Bias,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_q_bm,
            stride_k_bn,
            stride_vn,
            stride_mm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dq_bm,
            stride_dk_bn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            rankdim,
            ATOMIC_ADD=True,
            MASK_TYPE=MASK_TYPE,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_RANKDIM=BLOCK_RANKDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_RANKDIM=EVEN_RANKDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(q, k, v, q_bias, k_bias, mask=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_mask = mask is not None
    mask_type = "none"
    if has_mask:
        assert mask.dtype in [q.dtype, torch.float]
        assert mask.is_cuda
        assert mask.dim() == 4
        if mask.stride(-1) != 1:
            mask = mask.contiguous()
        if mask.shape[2:] == (1, seqlen_k):
            mask_type = "vector"
        elif mask.shape[2:] == (seqlen_q, seqlen_k):
            mask_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of mask must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        mask = mask.expand(batch, nheads, seqlen_q, seqlen_k)
    mask_strides = (mask.stride(0), mask.stride(1), mask.stride(2)) if has_mask else (0, 0, 0)

    has_bias = q_bias is not None
    if has_bias:
        bias_type = "orthogonal"
        _, _, _, rank = q_bias.shape
        q_bias = q_bias.expand(batch, seqlen_q, nheads, rank)
        k_bias = k_bias.expand(batch, seqlen_k, nheads, rank)
        BLOCK_RANKDIM = max(triton.next_power_of_2(rank), 16)
    else:
        bias_type = None
        rank = 0
        BLOCK_RANKDIM = 16
    q_bias_strides = (q_bias.stride(0), q_bias.stride(2), q_bias.stride(1)) if (q_bias is not None) else (0, 0, 0)
    k_bias_strides = (k_bias.stride(0), k_bias.stride(2), k_bias.stride(1)) if (k_bias is not None) else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        q_bias,
        k_bias,
        mask,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *q_bias_strides,
        *k_bias_strides,
        *mask_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        rank,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        mask_type,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_RANKDIM,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attn_backward(
        do, q, k, v, o, q_bias, k_bias, lse, dq, dk, dv, dq_bias, dk_bias, mask=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)

    if q_bias is not None:
        bias_type = "orth"
        assert dq_bias.stride(-1) == dk_bias.stride(-1)
        _, _, _, rank = q_bias.shape
        dq_bias_accum = torch.empty_like(q_bias, dtype=torch.float32)
        BLOCK_RANKDIM = max(triton.next_power_of_2(rank), 16)
    else:
        bias_type = None
        dq_bias_accum = None
        BLOCK_RANKDIM = 16
        rank = 0

    q_bias_strides = (q_bias.stride(0), q_bias.stride(2), q_bias.stride(1)) if (q_bias is not None) else (0, 0, 0)
    k_bias_strides = (k_bias.stride(0), k_bias.stride(2), k_bias.stride(1)) if (k_bias is not None) else (0, 0, 0)
    dq_bias_accum_strides = (q_bias.stride(0), q_bias.stride(2), q_bias.stride(1)) if (q_bias is not None) else (
        0, 0, 0)
    dk_bias_strides = (k_bias.stride(0), k_bias.stride(2), k_bias.stride(1)) if (k_bias is not None) else (0, 0, 0)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_mask = mask is not None
    mask_type = "none"
    if has_mask:
        assert mask.dtype in [q.dtype, torch.float]
        assert mask.is_cuda
        assert mask.dim() == 4
        assert mask.stride(-1) == 1
        if mask.shape[2:] == (1, seqlen_k):
            mask_type = "vector"
        elif mask.shape[2:] == (seqlen_q, seqlen_k):
            mask_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of mask must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        mask = mask.expand(batch, nheads, seqlen_q, seqlen_k)
    mask_strides = (mask.stride(0), mask.stride(1), mask.stride(2)) if has_mask else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        q_bias,
        k_bias,
        mask,
        do,
        dq_accum,
        dq_bias_accum,
        dk,
        dk_bias,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        *q_bias_strides,
        k.stride(0),
        k.stride(2),
        k.stride(1),
        *k_bias_strides,
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *mask_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        *dq_bias_accum_strides,
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        *dk_bias_strides,
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        rank,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        mask_type,
        bias_type,
        causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_RANKDIM=BLOCK_RANKDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)
    if q_bias is not None:
        dq_bias.copy_(dq_bias_accum)


class FlashBiasFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_bias=None, k_bias=None, mask=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        q_bias: (batch_size, seqlen_q, nheads, rankdim) or (1, seqlen_q, nheads, rankdim) or (1, seqlen_q, 1, rankdim) or (batch_size, seqlen_q, 1, rankdim)
        k_bias: (batch_size, seqlen_k, nheads, rankdim) or (1, seqlen_k, nheads, rankdim) or (1, seqlen_q, 1, rankdim) or (batch_size, seqlen_q, 1, rankdim)
        mask: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
        softmax_scale: should be set as 1 / sqrt(headdim), otherwise directly multiply to the q vector. If without any input, it will be set as 1 / sqrt(headdim)
        """
        # Make sure that the last dimension is contiguous
        if q_bias is not None:
            q, k, v, q_bias, k_bias = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v, q_bias, k_bias]]
        else:
            q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, q_bias, k_bias, mask=mask, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, q_bias, k_bias, lse, mask)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, q_bias, k_bias, lse, mask = ctx.saved_tensors
        assert not ctx.needs_input_grad[5], "FlashBias does not support mask gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            if q_bias is not None:
                dq_bias = torch.empty_like(q_bias)
                dk_bias = torch.empty_like(k_bias)
            else:
                dq_bias = None
                dk_bias = None
            _flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                q_bias,
                k_bias,
                lse,
                dq,
                dk,
                dv,
                dq_bias,
                dk_bias,
                mask=mask,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, dq_bias, dk_bias, None, None, None


flash_bias_func = FlashBiasFunc.apply
