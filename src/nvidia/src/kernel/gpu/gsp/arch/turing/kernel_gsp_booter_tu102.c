/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gpu/gsp/kernel_gsp.h"

#include "nv.h"
#include "os-interface.h"

#include "gpu/gpu.h"
#include "gpu/falcon/kernel_falcon.h"
#include "gpu/sec2/kernel_sec2.h"

#include "published/turing/tu102/dev_falcon_v4.h"
#include "published/turing/tu102/dev_fb.h"

#define CMP90_PC_EXACT_PCI_DEVICE_ID             0x220DU
#define CMP90_PC_EXACT_PCI_SUBDEVICE_ID          0x155510DEU
#define CMP90_PC_EXACT_FEAT_OVR_PLM              0x00823804U
#define CMP90_PC_EXACT_PLM_OPEN                  0xffffffffU
#define CMP90_PC_EXACT_TRACEIDX                   0x00000148U
#define CMP90_PC_EXACT_TRACEPC                    0x0000014CU
#define CMP90_PC_EXACT_SS0                       0x0082381cU
#define CMP90_PC_EXACT_SS1                       0x00823820U
#define CMP90_PC_EXACT_SS0_FULL                  0x88888888U
#define CMP90_PC_EXACT_SS1_FULL                  0x00000008U

NV_STATUS kgspCmp90RestoreStockSignatureForBooter(OBJGPU *pGpu,
                                                  KernelGsp *pKernelGsp);
NV_STATUS kgspExecuteBooterUnloadIfNeeded_TU102(OBJGPU *pGpu,
                                                KernelGsp *pKernelGsp,
                                                const NvU64 sysmemAddr);

#define CMP90_PC_EXACT_MAX_GPU_INSTANCES        8U

static NvBool
s_cmp90PcExactTimingAttemptedByGpu[CMP90_PC_EXACT_MAX_GPU_INSTANCES];
static NvBool
s_cmp90PcStockRetryArmedByGpu[CMP90_PC_EXACT_MAX_GPU_INSTANCES];

static NvU32
_kgspCmp90Rejoin14GpuSlot
(
    OBJGPU *pGpu
)
{
    return (pGpu->gpuInstance < CMP90_PC_EXACT_MAX_GPU_INSTANCES) ?
        pGpu->gpuInstance : 0U;
}

static NvBool
_kgspCmp90Rejoin10SelectorsFull
(
    OBJGPU *pGpu,
    NvU32 *pSs0,
    NvU32 *pSs1
)
{
    NvU32 ss0 = GPU_REG_RD32(pGpu, CMP90_PC_EXACT_SS0);
    NvU32 ss1 = GPU_REG_RD32(pGpu, CMP90_PC_EXACT_SS1);

    if (pSs0 != NULL)
        *pSs0 = ss0;
    if (pSs1 != NULL)
        *pSs1 = ss1;

    return ((ss0 == CMP90_PC_EXACT_SS0_FULL) &&
            (ss1 == CMP90_PC_EXACT_SS1_FULL));
}

static NV_STATUS
s_executeBooterUcode_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspFlcnUcode *pBooterUcode,
    KernelFalcon *pKernelFlcn,
    const NvU32 mailbox0Arg,
    const NvU32 mailbox1Arg,
    NvU32 *pMailbox0Result,
    NvU32 *pCpuCtlResult,
    NvU32 *pIrqStatResult,
    NvU32 *pDebugInfoResult,
    NvU32 *pTraceIdxResult,
    NvU32 *pTracePcResult
)
{
    NV_STATUS status;
    NvU32 mailbox0, mailbox1;

    NV_ASSERT_OR_RETURN(pBooterUcode != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pKernelFlcn != NULL, NV_ERR_INVALID_STATE);

    mailbox0 = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_MAILBOX0);
    mailbox1 = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_MAILBOX1);

    NV_PRINTF(LEVEL_INFO, "before Booter mailbox0 0x%08x, mailbox1 0x%08x\n", mailbox0, mailbox1);

    mailbox0 = mailbox0Arg;
    mailbox1 = mailbox1Arg;

    NV_PRINTF(LEVEL_INFO, "starting Booter with mailbox0 0x%08x, mailbox1 0x%08x\n", mailbox0, mailbox1);

    status = kgspExecuteHsFalcon_HAL(pGpu, pKernelGsp,
                                     pBooterUcode, pKernelFlcn,
                                     &mailbox0, &mailbox1);

    NV_PRINTF(LEVEL_INFO, "after Booter mailbox0 0x%08x, mailbox1 0x%08x\n", mailbox0, mailbox1);

    if (pMailbox0Result != NULL)
        *pMailbox0Result = mailbox0;
    if (pCpuCtlResult != NULL)
        *pCpuCtlResult = kflcnRegRead_HAL(
            pGpu, pKernelFlcn, NV_PFALCON_FALCON_CPUCTL);
    if (pIrqStatResult != NULL)
        *pIrqStatResult = kflcnRegRead_HAL(
            pGpu, pKernelFlcn, NV_PFALCON_FALCON_IRQSTAT);
    if (pDebugInfoResult != NULL)
        *pDebugInfoResult = kflcnRegRead_HAL(
            pGpu, pKernelFlcn, NV_PFALCON_FALCON_DEBUGINFO);
    if (pTraceIdxResult != NULL)
        *pTraceIdxResult = kflcnRegRead_HAL(
            pGpu, pKernelFlcn, CMP90_PC_EXACT_TRACEIDX);
    if (pTracePcResult != NULL)
        *pTracePcResult = kflcnRegRead_HAL(
            pGpu, pKernelFlcn, CMP90_PC_EXACT_TRACEPC);

    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "failed to execute Booter: status 0x%x, mailbox 0x%x\n", status, mailbox0);
        return status;
    }

    if (mailbox0 != 0)
    {
        NV_PRINTF(LEVEL_ERROR, "Booter failed with non-zero error code: 0x%x\n", mailbox0);
        return NV_ERR_GENERIC;
    }

    return status;
}

NV_STATUS
kgspExecuteBooterLoad_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    const NvU64 sysmemAddrOfData
)
{
    NV_STATUS status;
    NvU32 mailbox0 = 0, mailbox1 = 0;
    NvU32 mailbox0Result = 0xffffffffU;
    NvU32 cpuCtl = 0xffffffffU;
    NvU32 irqStat = 0xffffffffU;
    NvU32 debugInfo = 0xffffffffU;
    NvU32 traceIdx = 0xffffffffU;
    NvU32 tracePc = 0xffffffffU;
    NvU32 featPlmBefore = 0xffffffffU;
    NvU32 featPlmAfter = 0xffffffffU;
    NvBool bCmp90PcStockRetry = NV_FALSE;
    NvBool bCmp90PcCanary =
        (((pGpu->idInfo.PCIDeviceID >> 16) ==
            CMP90_PC_EXACT_PCI_DEVICE_ID) &&
         (pGpu->idInfo.PCISubDeviceID ==
            CMP90_PC_EXACT_PCI_SUBDEVICE_ID));
    NvU32 cmp90PcGpuSlot = bCmp90PcCanary ?
        _kgspCmp90Rejoin14GpuSlot(pGpu) : 0U;

    KernelSec2 *pKernelSec2 = GPU_GET_KERNEL_SEC2(pGpu);

    NV_ASSERT_OR_RETURN(pKernelGsp->pBooterLoadUcode != NULL, NV_ERR_INVALID_STATE);

    if (bCmp90PcCanary &&
        s_cmp90PcExactTimingAttemptedByGpu[cmp90PcGpuSlot])
    {
        NvU32 ss0Current = 0xffffffffU;
        NvU32 ss1Current = 0xffffffffU;

        if (_kgspCmp90Rejoin10SelectorsFull(
                pGpu, &ss0Current, &ss1Current))
        {
            bCmp90PcStockRetry = NV_TRUE;
            s_cmp90PcStockRetryArmedByGpu[cmp90PcGpuSlot] = NV_FALSE;
            NV_PRINTF(
                LEVEL_ERROR,
                "CMP90_STOCKFLOW_REJOIN10: allowing stock Booter after "
                "FLR with full selectors ss0=0x%08x ss1=0x%08x\n",
                ss0Current, ss1Current);
        }
        else if (s_cmp90PcStockRetryArmedByGpu[cmp90PcGpuSlot])
        {
            bCmp90PcStockRetry = NV_TRUE;
            s_cmp90PcStockRetryArmedByGpu[cmp90PcGpuSlot] = NV_FALSE;
            NV_PRINTF(
                LEVEL_ERROR,
                "CMP90_STOCKFLOW_REJOIN10: allowing one stock Booter "
                "retry after early PLM handoff\n");
        }
        else
        {
            NV_PRINTF(
                LEVEL_ERROR,
                "CMP90_PROD_STACK_SHIFT_PLM_V67: "
                "module-lifetime gate blocked "
                "an additional Booter Load request\n");
            return NV_ERR_INVALID_STATE;
        }
    }

    if (sysmemAddrOfData != 0)
    {
        //
        // sysmemAddrOfData either represents the FW WPR MetaData or the FW SR Data as a physical address in SYSTEM
        // Provide that data in falcon SEC mailboxes 0 (low 32 bits) and 1 (high 32 bits)
        //
        mailbox0 = NvU64_LO32(sysmemAddrOfData);
        mailbox1 = NvU64_HI32(sysmemAddrOfData);
    }

    NV_PRINTF(LEVEL_INFO, "executing Booter Load, sysmemAddrOfData 0x%llx\n",
              sysmemAddrOfData);

    if (bCmp90PcCanary)
    {
        s_cmp90PcExactTimingAttemptedByGpu[cmp90PcGpuSlot] = NV_TRUE;
        featPlmBefore = GPU_REG_RD32(
            pGpu, CMP90_PC_EXACT_FEAT_OVR_PLM);
    }

    NV_ASSERT_OK_OR_RETURN(kflcnReset_HAL(pGpu, staticCast(pKernelSec2, KernelFalcon)));

    status = s_executeBooterUcode_TU102(pGpu, pKernelGsp,
                                        pKernelGsp->pBooterLoadUcode,
                                        staticCast(pKernelSec2, KernelFalcon),
                                        mailbox0, mailbox1,
                                        bCmp90PcCanary ?
                                            &mailbox0Result : NULL,
                                        bCmp90PcCanary ? &cpuCtl : NULL,
                                        bCmp90PcCanary ? &irqStat : NULL,
                                        bCmp90PcCanary ? &debugInfo : NULL,
                                        bCmp90PcCanary ? &traceIdx : NULL,
                                        bCmp90PcCanary ? &tracePc : NULL);
    if (bCmp90PcCanary)
    {
        featPlmAfter = GPU_REG_RD32(
            pGpu, CMP90_PC_EXACT_FEAT_OVR_PLM);
        NV_PRINTF(
            LEVEL_ERROR,
            "CMP90_PROD_STACK_SHIFT_PLM_V67: native GA102 "
            "status=0x%x mailbox0=0x%08x "
            "FEAT_PLM_before=0x%08x FEAT_PLM_after=0x%08x "
            "cpu_ctl=0x%08x irq_stat=0x%08x "
            "debug_info=0x%08x trace_idx=0x%08x trace_pc=0x%08x "
            "memdesc=%llu wpr_meta=%llu; "
            "CMP90_STOCKFLOW_REJOIN9 continuing into stock "
            "GSP-RM continuation\n",
            status, mailbox0Result, featPlmBefore, featPlmAfter,
            cpuCtl, irqStat, debugInfo, traceIdx, tracePc,
            (unsigned long long)memdescGetSize(
                pKernelGsp->pSignatureMemdesc),
            (unsigned long long)((pKernelGsp->pWprMetaV1 != NULL) ?
                pKernelGsp->pWprMetaV1->sizeOfSignature : 0));
        if ((status != NV_OK) &&
            (featPlmAfter == CMP90_PC_EXACT_PLM_OPEN) &&
            !bCmp90PcStockRetry)
        {
            NV_STATUS restoreStatus =
                kgspCmp90RestoreStockSignatureForBooter(pGpu, pKernelGsp);
            NvU32 ss0Before;
            NvU32 ss1Before;
            NvU32 ss0After;
            NvU32 ss1After;
            NvU32 wpr2LoBefore;
            NvU32 wpr2HiBefore;
            NvU32 wpr2LoAfter;
            NvU32 wpr2HiAfter;

            ss0Before = GPU_REG_RD32(pGpu, CMP90_PC_EXACT_SS0);
            ss1Before = GPU_REG_RD32(pGpu, CMP90_PC_EXACT_SS1);
            wpr2LoBefore = GPU_REG_RD32(
                pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
            wpr2HiBefore = GPU_REG_RD32(
                pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);

            GPU_REG_WR32(pGpu, CMP90_PC_EXACT_SS1,
                         CMP90_PC_EXACT_SS1_FULL);
            GPU_REG_WR32(pGpu, CMP90_PC_EXACT_SS0,
                         CMP90_PC_EXACT_SS0_FULL);

            ss0After = GPU_REG_RD32(pGpu, CMP90_PC_EXACT_SS0);
            ss1After = GPU_REG_RD32(pGpu, CMP90_PC_EXACT_SS1);
            wpr2LoAfter = GPU_REG_RD32(
                pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
            wpr2HiAfter = GPU_REG_RD32(
                pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
            featPlmAfter = GPU_REG_RD32(
                pGpu, CMP90_PC_EXACT_FEAT_OVR_PLM);

            if (restoreStatus != NV_OK)
            {
                NV_PRINTF(
                    LEVEL_ERROR,
                    "CMP90_STOCKFLOW_REJOIN10: failed to restore stock "
                    "signature before FLR status=0x%x\n",
                    restoreStatus);
                return restoreStatus;
            }

            {
                nv_state_t *nv = NV_GET_NV_STATE(pGpu);

                if ((nv == NULL) || (nv->handle == NULL))
                {
                    NV_PRINTF(
                        LEVEL_ERROR,
                        "CMP90_STOCKFLOW_REJOIN11: cannot trigger PCIe "
                        "FLR because nv_state/handle is missing nv=%p\n",
                        nv);
                    return NV_ERR_INVALID_STATE;
                }

                nv->flags |= NV_FLAG_TRIGGER_FLR;
                NV_PRINTF(
                    LEVEL_ERROR,
                    "CMP90_STOCKFLOW_REJOIN12: armed official PCIe FLR "
                    "on init-failure cleanup "
                    "handle=%p flags=0x%x\n",
                    nv->handle, nv->flags);
            }
            s_cmp90PcStockRetryArmedByGpu[cmp90PcGpuSlot] = NV_TRUE;

            NV_PRINTF(
                LEVEL_ERROR,
                "CMP90_STOCKFLOW_REJOIN12: wrote full-speed selectors "
                "and armed official PCIe FLR ss0_before=0x%08x ss1_before=0x%08x "
                "ss0_after=0x%08x ss1_after=0x%08x "
                "wpr2_lo_before=0x%08x wpr2_hi_before=0x%08x "
                "wpr2_lo_after=0x%08x wpr2_hi_after=0x%08x "
                "wpr2_up=%u feat_plm_after=0x%08x "
                "memdesc=%llu wpr_meta=%llu; returning reset-required\n",
                ss0Before, ss1Before, ss0After, ss1After,
                wpr2LoBefore, wpr2HiBefore,
                wpr2LoAfter, wpr2HiAfter,
                kgspIsWpr2Up_HAL(pGpu, pKernelGsp),
                featPlmAfter,
                (unsigned long long)memdescGetSize(
                    pKernelGsp->pSignatureMemdesc),
                (unsigned long long)((pKernelGsp->pWprMetaV1 != NULL) ?
                    pKernelGsp->pWprMetaV1->sizeOfSignature : 0));

            NV_PRINTF(LEVEL_ERROR,
                "CMP90_STOCKFLOW_REJOIN12: returning reset-required-no-internal-retry\n");

            return NV_ERR_RESET_REQUIRED;
        }
    }
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "failed to execute Booter Load: 0x%x\n", status);
        return status;
    }

    return status;
}

NV_STATUS
kgspExecuteBooterUnloadIfNeeded_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    const NvU64 sysmemAddrOfSuspendResumeData
)
{
    NV_STATUS status;
    KernelSec2 *pKernelSec2 = GPU_GET_KERNEL_SEC2(pGpu);
    NvU32 mailbox0 = 0xFF, mailbox1 = 0xFF;

    if (IS_GPU_GC6_STATE_ENTERING(pGpu))
    {
        mailbox0 = mailbox1 = 0xdeaddead;
    }

    if (API_GPU_IN_RESET_SANITY_CHECK(pGpu))
        return NV_ERR_GPU_IN_FULLCHIP_RESET;

    // skip actually executing Booter Unload if WPR2 is not up
    if (!kgspIsWpr2Up_HAL(pGpu, pKernelGsp))
    {
        NV_PRINTF(LEVEL_INFO, "skipping executing Booter Unload as WPR2 is not up\n");
        return NV_OK;
    }

    NV_PRINTF(LEVEL_INFO, "executing Booter Unload\n");
    NV_ASSERT_OR_RETURN(pKernelGsp->pBooterUnloadUcode != NULL, NV_ERR_INVALID_STATE);

    NV_ASSERT_OK(kflcnReset_HAL(pGpu, staticCast(pKernelSec2, KernelFalcon)));

    // SR code
    if (sysmemAddrOfSuspendResumeData != 0)
    {
        mailbox0 = NvU64_LO32(sysmemAddrOfSuspendResumeData);
        mailbox1 = NvU64_HI32(sysmemAddrOfSuspendResumeData);
    }
    status = s_executeBooterUcode_TU102(pGpu, pKernelGsp,
                                        pKernelGsp->pBooterUnloadUcode,
                                        staticCast(pKernelSec2, KernelFalcon),
                                        mailbox0, mailbox1, NULL,
                                        NULL, NULL, NULL, NULL, NULL);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "failed to execute Booter Unload: 0x%x\n", status);
        return status;
    }

    if (IS_GPU_GC6_STATE_ENTERING(pGpu))
    {
        // For GC6 path, WPR2 should still be up (not torn down)
        if (!kgspIsWpr2Up_HAL(pGpu, pKernelGsp))
        {
            NV_PRINTF(LEVEL_ERROR, "failed to execute Booter Unload: WPR2 is cleared despite GC6\n");
            return NV_ERR_GENERIC;
        }
    }
    else
    {
        // For all other unloads (non-GC6), WPR2 should be torn down
        if (kgspIsWpr2Up_HAL(pGpu, pKernelGsp))
        {
            NV_PRINTF(LEVEL_ERROR, "failed to execute Booter Unload: WPR2 is still up\n");
            return NV_ERR_GENERIC;
        }
    }

    return status;
}
