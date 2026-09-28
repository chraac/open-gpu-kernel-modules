/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*!
 * Provides TU102+ specific KernelGsp HAL implementations.
 */

#include "gpu/gsp/kernel_gsp.h"
#include "gpu/gsp/gsp_init_args.h"

#include "gpu/rc/kernel_rc.h"
#include "gpu/disp/kern_disp.h"
#include "gpu/mem_mgr/mem_mgr.h"
#include "gpu/mem_sys/kern_mem_sys.h"
#include "vgpu/rpc.h"
#include "core/thread_state.h"
#include "os/os.h"
#include "nverror.h"
#include "nvrm_registry.h"
#include "crashcat/crashcat_report.h"

#include "published/turing/tu102/dev_gsp.h"

#define CMP50_PC_CANARY_PCI_DEVICE_ID       0x1E0910DEU
#define CMP50_PC_CANARY_PCI_SUBDEVICE_MATCHES(id) (((id) == 0x155410DEU) || ((id) == 0x371F1462U))
#define CMP50_FECS_FEATURE_PLM               0x00409650U
#define CMP50_FECS_FEATURE_PLM_POSTLOCK      0xFFFFFF8FU
#define CMP50_FECS_SM_SPEED_OVERRIDE_0       0x00409664U
#define CMP50_FECS_SM_SPEED_OVERRIDE_1       0x0040966CU
#define CMP50_FECS_SM_SPEED_FULL_0           0x88888888U
#define CMP50_FECS_SM_SPEED_FULL_1           0x00000008U
#define CMP50_FECS_FEATURE_READOUT           0x00409660U
#define CMP50_FECS_FEATURE_OVERRIDE_SM_SPEED_SELECT 0x00409664U
#define CMP50_FECS_FEATURE_READOUT_SM_SPEED_SELECT  0x00409668U
#define CMP50_FECS_FEATURE_OVERRIDE_SM_SPEED_SELECT_1 0x0040966CU
#define CMP50_WPR2_STOCK_SPAN                0x00000E00U
#define CMP50_WPR2_ADDR_LO_DOWN              0x1FFFFE00U
#define CMP50_SEC2_RESET_PLM                  0x008403C4U
#define CMP50_SEC2_RESET_PLM_STOCK            0x0000008FU
#define CMP50_PCIE_LINK_CAP                  0x00088084U
#define CMP50_PCIE_LINK_CAP2                 0x000880A4U
#define CMP50_PCIE_LINK_CTRL2                0x000880A8U
#define CMP50_PCIE_LINK_STATUS               0x00088088U
#define CMP50_PCIE_VSEC_DEVICE               0x0008860CU
#define CMP50_PCIE_VSEC_HIERARCHY            0x00088610U
#define CMP50_PCIE_LTSSM                     0x0008872CU
#define CMP50_PCIE_PRIV_MISC_1               0x0008841CU
#define CMP50_PCIE_PRIV_MISC_1_GEN2_EN       ((1U << 11) | (1U << 13))
#define CMP50_PCIE_PRIV_MISC_1_GEN2_VAL      ((1U << 12) | (1U << 14))
#define CMP50_PCIE_LINK_CONFIG0              0x0008C040U
#define CMP50_PCIE_PL_LINK_RATE              0x0008C1C0U
#define CMP50_PCIE_CYA0                      0x0008C2C0U
#define CMP50_PCIE_XP3G_STATUS0              0x0008E100U
#define CMP50_PCIE_XP3G_OVERRIDE0            0x0008E110U
#define CMP50_PCIE_XP3G_VALUE0               0x0008E120U
#define CMP50_PCIE_XP3G_STATUS3              0x0008E10CU
#define CMP50_PCIE_XP3G_OVERRIDE3            0x0008E11CU
#define CMP50_PCIE_XP3G_VALUE3               0x0008E12CU
#define CMP50_PCIE_XP3G_PLM0                 0x0008E1B0U
#define CMP50_PCIE_PL_LINK_RATE_SPEED_MASK   0x00060000U
#define CMP50_PCIE_PL_LINK_RATE_GEN2         0x00040000U

void kgspCmp50SetExploitMode(OBJGPU *pGpu, NvBool bEnabled);
NV_STATUS kgspCmp50RebuildStockSignature(OBJGPU *pGpu, KernelGsp *pKernelGsp);
NV_STATUS kgspCmp50SanitizeSec2AfterExploit(OBJGPU *pGpu,
                                            KernelGsp *pKernelGsp);
NV_STATUS kgspCmp50ReplaceSignature(OBJGPU *pGpu, KernelGsp *pKernelGsp,
                                    NvBool bSecondExploit);

// Keep RM's automatic init retry from rebuilding FRTS/WPR after the bounded
// V140 handoff transaction has committed. This is deliberately module-lifetime
// only and keyed by stable PCI identity: RM can recycle gpuInstance values
// across retry attempts in a multi-card module probe or boot.
#define CMP50_STOCKFLOW_BDF_SLOTS 8192U
static NvBool g_cmp50V274PhaseCommitted[CMP50_STOCKFLOW_BDF_SLOTS] = { NV_FALSE };
static NvBool g_cmp50V274FwsecReady[CMP50_STOCKFLOW_BDF_SLOTS] = { NV_FALSE };
static NvBool g_cmp50V274GspReady[CMP50_STOCKFLOW_BDF_SLOTS] = { NV_FALSE };
static NvU32 g_cmp50StockWprLo[CMP50_STOCKFLOW_BDF_SLOTS] = { 0U };
static NvU32 g_cmp50StockWprHi[CMP50_STOCKFLOW_BDF_SLOTS] = { 0U };

static NvBool
s_isCmp50ComputeUnlock
(
    OBJGPU *pGpu
)
{
    return (pGpu->idInfo.PCIDeviceID == CMP50_PC_CANARY_PCI_DEVICE_ID) &&
           CMP50_PC_CANARY_PCI_SUBDEVICE_MATCHES(pGpu->idInfo.PCISubDeviceID);
}

static void
s_cmp50ApplyGen2Policy
(
    OBJGPU *pGpu,
    const char *phase
)
{
    NvU32 xp3gPlm = GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_PLM0);
    NvU32 vsecDevice = GPU_REG_RD32(pGpu, CMP50_PCIE_VSEC_DEVICE);
    NvU32 privMisc = GPU_REG_RD32(pGpu, CMP50_PCIE_PRIV_MISC_1);
    NvU32 privMiscWant =
        (privMisc | CMP50_PCIE_PRIV_MISC_1_GEN2_EN) &
        ~CMP50_PCIE_PRIV_MISC_1_GEN2_VAL;
    NvU32 origOvr0;
    NvU32 origVal0;
    NvU32 origOvr3;
    NvU32 origVal3;
    NvU32 origHierarchy;
    NvU32 origCya0;
    NvU32 origLinkConfig0;
    NvU32 origPlLinkRate;
    NvU32 origLtssm;
    NvBool policyOk;

    if (xp3gPlm != 0xFFFFFFFFU)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_GEN2: PROTECTED_GATE_FAIL phase=%s "
                  "XP3G_PLM=%08x VSEC=%08x PRIV_MISC=%08x\n",
                  phase, xp3gPlm, vsecDevice, privMisc);
        return;
    }

    origOvr0 = GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0);
    origVal0 = GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE0);
    origOvr3 = GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3);
    origVal3 = GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE3);
    origHierarchy = GPU_REG_RD32(pGpu, CMP50_PCIE_VSEC_HIERARCHY);
    origCya0 = GPU_REG_RD32(pGpu, CMP50_PCIE_CYA0);
    origLinkConfig0 = GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CONFIG0);
    origPlLinkRate = GPU_REG_RD32(pGpu, CMP50_PCIE_PL_LINK_RATE);
    origLtssm = GPU_REG_RD32(pGpu, CMP50_PCIE_LTSSM);

    /*
     * The capability adoption is one-shot: the LTSSM=6 kick below makes the
     * card regenerate its PCIe config block from whatever the policy regs
     * hold at that instant. This function runs twice (post-booter and
     * gsp-ready). On a cold boot the post-booter pass adopts Gen2, then GSP
     * starts clobbering the policy regs, and a second kick at gsp-ready
     * re-derives the now-locked set and destroys the Gen2 we just won
     * (192.168.1.224, 2026-09-17: post-booter CAP=453d02 -> gsp-ready
     * CAP=453d01). Boots that came up already Gen2 never hit this because
     * GSP does not fight an already-unlocked block. So once the capability
     * reads Gen2, do not touch the policy again - only keep the target link
     * speed latched and return.
     */
    if ((GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CAP) & 0xFU) >= 2U)
    {
        NvU32 linkCtrl2 = GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CTRL2);

        if ((linkCtrl2 & 0xFU) != 2U)
            GPU_REG_WR32(pGpu, CMP50_PCIE_LINK_CTRL2,
                         (linkCtrl2 & ~0xFU) | 2U);
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_GEN2: ALREADY_GEN2 phase=%s CAP=%08x LC2=%08x "
                  "(not re-kicking)\n",
                  phase, GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CAP),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CTRL2));
        return;
    }

    GPU_REG_WR32(pGpu, CMP50_PCIE_PRIV_MISC_1, privMiscWant);
    GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_VALUE0, 0U);
    GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0, 1U);
    GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_VALUE3, 0x00200000U);
    GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3, 4U);
    GPU_REG_WR32(pGpu, CMP50_PCIE_VSEC_HIERARCHY,
                 (origHierarchy & ~(1U << 12)) | 1U);
    GPU_REG_WR32(pGpu, CMP50_PCIE_CYA0, origCya0 & ~(1U << 2));
    GPU_REG_WR32(pGpu, CMP50_PCIE_LINK_CONFIG0,
                 (origLinkConfig0 & ~0x000C0000U) | (2U << 18));
    GPU_REG_WR32(pGpu, CMP50_PCIE_PL_LINK_RATE,
                 (origPlLinkRate & ~CMP50_PCIE_PL_LINK_RATE_SPEED_MASK) |
                     CMP50_PCIE_PL_LINK_RATE_GEN2);
    GPU_REG_WR32(pGpu, CMP50_PCIE_LTSSM, 6U);

    policyOk =
        (GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE0) == 0U) &&
        (GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0) == 1U) &&
        (GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE3) == 0x00200000U) &&
        (GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3) == 4U) &&
        ((GPU_REG_RD32(pGpu, CMP50_PCIE_PRIV_MISC_1) &
          CMP50_PCIE_PRIV_MISC_1_GEN2_EN) ==
            CMP50_PCIE_PRIV_MISC_1_GEN2_EN) &&
        ((GPU_REG_RD32(pGpu, CMP50_PCIE_PRIV_MISC_1) &
          CMP50_PCIE_PRIV_MISC_1_GEN2_VAL) == 0U) &&
        ((GPU_REG_RD32(pGpu, CMP50_PCIE_VSEC_HIERARCHY) &
          ((1U << 12) | 1U)) == 1U) &&
        ((GPU_REG_RD32(pGpu, CMP50_PCIE_CYA0) & (1U << 2)) == 0U) &&
        (((GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CONFIG0) >> 18) & 3U) == 2U) &&
        ((GPU_REG_RD32(pGpu, CMP50_PCIE_PL_LINK_RATE) &
            CMP50_PCIE_PL_LINK_RATE_SPEED_MASK) ==
            CMP50_PCIE_PL_LINK_RATE_GEN2) &&
        (GPU_REG_RD32(pGpu, CMP50_PCIE_LTSSM) == 6U);

    if (!policyOk)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_GEN2: POLICY_MISMATCH phase=%s "
                  "OVR=%08x/%08x VAL=%08x/%08x HIER=%08x "
                  "PRIV=%08x LC2=%08x CYA=%08x CFG=%08x PL=%08x "
                  "LTSSM=%08x\n",
                  phase,
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE3),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_VSEC_HIERARCHY),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_PRIV_MISC_1),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CTRL2),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_CYA0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CONFIG0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_PL_LINK_RATE),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LTSSM));
        GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0, origOvr0);
        GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_VALUE0, origVal0);
        GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3, origOvr3);
        GPU_REG_WR32(pGpu, CMP50_PCIE_XP3G_VALUE3, origVal3);
        GPU_REG_WR32(pGpu, CMP50_PCIE_PRIV_MISC_1, privMisc);
        GPU_REG_WR32(pGpu, CMP50_PCIE_VSEC_HIERARCHY, origHierarchy);
        GPU_REG_WR32(pGpu, CMP50_PCIE_CYA0, origCya0);
        GPU_REG_WR32(pGpu, CMP50_PCIE_LINK_CONFIG0, origLinkConfig0);
        GPU_REG_WR32(pGpu, CMP50_PCIE_PL_LINK_RATE, origPlLinkRate);
        GPU_REG_WR32(pGpu, CMP50_PCIE_LTSSM, origLtssm);
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_GEN2: POLICY_ROLLBACK phase=%s "
                  "OVR=%08x/%08x VAL=%08x/%08x LC2=%08x "
                  "CFG=%08x PL=%08x LTSSM=%08x\n",
                  phase,
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_VALUE3),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CTRL2),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CONFIG0),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_PL_LINK_RATE),
                  GPU_REG_RD32(pGpu, CMP50_PCIE_LTSSM));
        return;
    }

    /*
     * Latch the Gen2 target link speed while the card still advertises the
     * unlocked capability: this is the only window where the write is not
     * dropped (with CAP=0x453d01 both the config register and this BAR0
     * mirror silently ignore it). Boots that came up with LC2 already at 2 -
     * inherited through standby power from a boot that had reached Gen2 -
     * kept CAP=0x453d02 through gsp-ready and adopted; boots that came up
     * cold with LC2=1 had the capability reverted by GSP and never adopted
     * again (journal of 192.168.1.224, boots of 2026-09-13/14 vs
     * 2026-09-17). Seeding it here is what restarts that chain.
     */
    if ((GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CAP) & 0xFU) >= 2U)
    {
        NvU32 linkCtrl2 = GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CTRL2);

        if ((linkCtrl2 & 0xFU) != 2U)
        {
            GPU_REG_WR32(pGpu, CMP50_PCIE_LINK_CTRL2,
                         (linkCtrl2 & ~0xFU) | 2U);
        }
    }

    NV_PRINTF(LEVEL_ERROR,
              "CMP50_GEN2: POLICY_PASS phase=%s CAP=%08x CAP2=%08x "
              "STAT=%08x XP3G=%08x/%08x/%08x/%08x "
              "VSEC=%08x/%08x PRIV=%08x LC2=%08x CFG=%08x PL=%08x\n",
              phase,
              GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CAP),
              GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CAP2),
              GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_STATUS),
              GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_STATUS0),
              GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE0),
              GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_STATUS3),
              GPU_REG_RD32(pGpu, CMP50_PCIE_XP3G_OVERRIDE3),
              GPU_REG_RD32(pGpu, CMP50_PCIE_VSEC_DEVICE),
              GPU_REG_RD32(pGpu, CMP50_PCIE_VSEC_HIERARCHY),
              GPU_REG_RD32(pGpu, CMP50_PCIE_PRIV_MISC_1),
              GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CTRL2),
              GPU_REG_RD32(pGpu, CMP50_PCIE_LINK_CONFIG0),
              GPU_REG_RD32(pGpu, CMP50_PCIE_PL_LINK_RATE));
}

static NvU32
s_cmp50StockflowGpuIndex
(
    OBJGPU *pGpu
)
{
    NvU32 slot = ((NvU32)gpuGetBus(pGpu) * 32U) +
                 (NvU32)gpuGetDevice(pGpu);

    if (slot < CMP50_STOCKFLOW_BDF_SLOTS)
        return slot;
    return pGpu->gpuInstance % CMP50_STOCKFLOW_BDF_SLOTS;
}
#include "published/turing/tu102/dev_gsp_addendum.h"
#include "published/turing/tu102/dev_riscv_pri.h"
#include "published/turing/tu102/dev_fbif_v4.h"
#include "published/turing/tu102/dev_falcon_v4.h"
#include "published/turing/tu102/dev_fb.h"  // for NV_PFB_PRI_MMU_WPR2_ADDR_HI
#include "published/turing/tu102/dev_fuse.h"
#include "published/turing/tu102/dev_ram.h"
#include "published/turing/tu102/dev_gc6_island.h"
#include "published/turing/tu102/dev_gc6_island_addendum.h"

#include "gpu/sec2/kernel_sec2.h"

#include "gpu/conf_compute/conf_compute.h"

#include "gpu/oob/kernel_oob.h"

#include "g_all_dcl_pb.h"
#include "lib/protobuf/prb.h"

#define SEC2_DEBUG_PRI_FEATURE_OVERRIDE_PLM         0x00823804
#define SEC2_DEBUG_PRI_FEATURE_OVERRIDE_SM_SPEED    0x0082381c
#define SEC2_DEBUG_PRI_FEATURE_OVERRIDE_SM_SPEED_1  0x00823820
#define SEC2_DEBUG_PRI_FBPA_CFG1                    0x009a0204
#define SEC2_DEBUG_PRI_MMU_LMR                      0x00100ce0

#define CMP90_PC_EXACT_PCI_DEVICE_ID                0x220DU
#define CMP90_PC_EXACT_PCI_DEVICE_ID_FULL           0x220D10DEU
#define CMP90_PC_EXACT_PCI_SUBDEVICE_ID             0x155510DEU

#include "events/gpu/ras/ras_events.h"
#include "nvoc/event_bus.h"

void
kgspConfigureFalcon_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    KernelFalconEngineConfig falconConfig;

    portMemSet(&falconConfig, 0, sizeof(falconConfig));

    falconConfig.registerBase       = DRF_BASE(NV_PGSP);
    falconConfig.riscvRegisterBase  = NV_FALCON2_GSP_BASE;
    falconConfig.fbifBase           = NV_PGSP_FBIF_BASE;
    falconConfig.bBootFromHs        = NV_FALSE;
    falconConfig.pmcEnableMask      = 0;
    falconConfig.bIsPmcDeviceEngine = NV_FALSE;
    falconConfig.physEngDesc        = ENG_GSP;

    // Enable CrashCat monitoring
    falconConfig.crashcatEngConfig.bEnable = NV_TRUE;
    falconConfig.crashcatEngConfig.pName = MAKE_NV_PRINTF_STR("GSP");
    falconConfig.crashcatEngConfig.errorId = GSP_ERROR;

    kflcnConfigureEngine(pGpu, staticCast(pKernelGsp, KernelFalcon), &falconConfig);
}

/*!
 * Check if the GSP is in debug mode
 *
 * @return whether the GSP is in debug mode or not
 */
NvBool
kgspIsDebugModeEnabled_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU32 data;

    data = GPU_REG_RD32(pGpu, NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS);

    return FLD_TEST_DRF(_FUSE, _OPT_SECURE_GSP_DEBUG_DIS, _DATA, _NO, data);
}

NV_STATUS
kgspAllocBootArgs_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvP64 pVa = NvP64_NULL;
    NvP64 pPriv = NvP64_NULL;
    NV_STATUS nvStatus = NV_OK;
    NvU64 flags = MEMDESC_FLAGS_NONE;

    if (confComputeForceUnprotAlloc(pGpu))
    {
        flags |= MEMDESC_FLAGS_ALLOC_IN_UNPROTECTED_MEMORY;
    }

    // Allocate WPR meta data
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescCreate(&pKernelGsp->pWprMetaV1Descriptor,
                                       pGpu, 0x1000, 0x1000,
                                       NV_TRUE, ADDR_SYSMEM, NV_MEMORY_CACHED,
                                       flags),
                        _kgspAllocBootArgs_exit_cleanup);

    memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_WPR_METADATA,
                    pKernelGsp->pWprMetaV1Descriptor);
    NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                         _kgspAllocBootArgs_exit_cleanup);

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescMap(pKernelGsp->pWprMetaV1Descriptor, 0,
                                    memdescGetSize(pKernelGsp->pWprMetaV1Descriptor),
                                    NV_TRUE, NV_PROTECT_READ_WRITE,
                                    &pVa, &pPriv),
                         _kgspAllocBootArgs_exit_cleanup);

    pKernelGsp->pWprMetaV1 = (GspFwWprMetaV1 *)NvP64_VALUE(pVa);
    pKernelGsp->pWprMetaV1MappingPriv = pPriv;

    portMemSet(pKernelGsp->pWprMetaV1, 0, sizeof(*pKernelGsp->pWprMetaV1));

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         kgspAllocBootArgsCommon(pGpu, pKernelGsp, flags),
                         _kgspAllocBootArgs_exit_cleanup);

    return nvStatus;

_kgspAllocBootArgs_exit_cleanup:
    kgspFreeBootArgs_HAL(pGpu, pKernelGsp);
    return nvStatus;
}

void
kgspFreeBootArgs_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    // release wpr meta data resources
    if (pKernelGsp->pWprMetaV1 != NULL)
    {
        memdescUnmap(pKernelGsp->pWprMetaV1Descriptor,
                     NV_TRUE,
                     (void *)pKernelGsp->pWprMetaV1,
                     pKernelGsp->pWprMetaV1MappingPriv);
        pKernelGsp->pWprMetaV1 = NULL;
        pKernelGsp->pWprMetaV1MappingPriv = NULL;
    }
    if (pKernelGsp->pWprMetaV1Descriptor != NULL)
    {
        memdescFree(pKernelGsp->pWprMetaV1Descriptor);
        memdescDestroy(pKernelGsp->pWprMetaV1Descriptor);
        pKernelGsp->pWprMetaV1Descriptor = NULL;
    }

    kgspFreeBootArgsCommon(pGpu, pKernelGsp);

    // Release radix3 version of GSP-RM ucode
    if (pKernelGsp->pGspUCodeRadix3Descriptor != NULL)
    {
        memdescFree(pKernelGsp->pGspUCodeRadix3Descriptor);
        memdescDestroy(pKernelGsp->pGspUCodeRadix3Descriptor);
        pKernelGsp->pGspUCodeRadix3Descriptor = NULL;
    }

    // Release signature memory
    if (pKernelGsp->pSignatureMemdesc != NULL)
    {
        memdescFree(pKernelGsp->pSignatureMemdesc);
        memdescDestroy(pKernelGsp->pSignatureMemdesc);
        pKernelGsp->pSignatureMemdesc = NULL;
    }
    if (pKernelGsp->pStockSignatureData != NULL)
    {
        portMemFree(pKernelGsp->pStockSignatureData);
        pKernelGsp->pStockSignatureData = NULL;
        pKernelGsp->stockSignatureSize = 0;
    }

    //
    // This should already have been freed after INIT_DONE, but maybe we errored
    // out earlier and need to do cleanup..
    //
    if (pKernelGsp->pExternalBindata != NULL)
    {
        memdescFree(pKernelGsp->pExternalBindata);
        memdescDestroy(pKernelGsp->pExternalBindata);
        pKernelGsp->pExternalBindata = NULL;
    }
}

// For LibOS2 this is called LIBOS_INTERRUPT_PROCESSOR_SUSPENDED
// For LibOS3 it got renamed to the #define we use below but the binary
// values are identical and the usage is the same.
#define INTERRUPT_PROCESSOR_SUSPENDED_VALUE 0x80000000

static NvBool
_kgspIsProcessorSuspended
(
    OBJGPU  *pGpu,
    void    *pVoid
)
{
    KernelGsp *pKernelGsp = reinterpretCast(pVoid, KernelGsp *);
    NvU32 mailbox;

    // Check for LIBOS_INTERRUPT_PROCESSOR_SUSPENDED in mailbox
    mailbox = kflcnRegRead_HAL(pGpu, staticCast(pKernelGsp, KernelFalcon),
                               NV_PFALCON_FALCON_MAILBOX0);
    return (mailbox & INTERRUPT_PROCESSOR_SUSPENDED_VALUE) != 0;
}

NV_STATUS
kgspWaitForProcessorSuspend_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp,
    NvBool     bProcessRpcs
)
{
    if (bProcessRpcs)
    {
        return gpuRpcConditionWait(pGpu, _kgspIsProcessorSuspended, pKernelGsp);
    }
    else
    {
        return gpuTimeoutCondWait(pGpu, _kgspIsProcessorSuspended, pKernelGsp, NULL);
    }
}

/*!
 * Load entrypoint address of boot binary into mailbox regs.
 */
void
kgspProgramLibosBootArgsAddr_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU64 addr =
        memdescGetPhysAddr(pKernelGsp->pLibosInitArgumentsDescriptor, AT_GPU, 0);

    GPU_REG_WR32(pGpu, NV_PGSP_FALCON_MAILBOX0, NvU64_LO32(addr));
    GPU_REG_WR32(pGpu, NV_PGSP_FALCON_MAILBOX1, NvU64_HI32(addr));
}

/*!
 * Prepare to boot GSP-RM
 *
 * This routine handles the prerequisites to booting GSP-RM that requires the API LOCK:
 *   - prepares boot binary image
 *   - prepares RISCV core to run GSP-RM
 *
 * Note that boot binary and GSP-RM images have already been placed
 * in the appropriate places by kgspPopulateWprMeta_HAL().
 *
 * Note that this routine is based on flcnBootstrapRiscvOS_GA102().
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      GSP object pointer
 * @param[in]   bootMode        GSP boot mode
 *
 * @return NV_OK if GSP-RM RISCV boot was successful.
 *         Appropriate NV_ERR_xxx value otherwise.
 */
NV_STATUS
kgspPrepareForBootstrap_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    NV_STATUS     status;
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);

    // Only for GSP client builds
    if (!IS_GSP_CLIENT(pGpu))
    {
        NV_PRINTF(LEVEL_ERROR, "IS_GSP_CLIENT is not set.\n");
        return NV_ERR_NOT_SUPPORTED;
    }

    if (!kflcnIsRiscvCpuEnabled_HAL(pGpu, pKernelFalcon))
    {
        NV_PRINTF(LEVEL_ERROR, "RISC-V core is not enabled.\n");
        return NV_ERR_NOT_SUPPORTED;
    }

    //
    // Prepare to execute FWSEC to setup FRTS if we have a FRTS region
    // Note: for resume and GC6 exit, FRTS is restored by Booter not FWSEC
    //
    if ((bootMode == KGSP_BOOT_MODE_NORMAL) &&
        (kgspGetFrtsSize_HAL(pGpu, pKernelGsp) > 0))
    {
        pKernelGsp->pPreparedFwsecCmd = portMemAllocNonPaged(sizeof(KernelGspPreparedFwsecCmd));
        status = kgspPrepareForFwsecFrts_HAL(pGpu, pKernelGsp,
                                             pKernelGsp->pFwsecUcode,
                                             pKernelGsp->pWprMetaV1->frtsOffset,
                                             pKernelGsp->pPreparedFwsecCmd);
        if (status != NV_OK)
        {
            portMemFree(pKernelGsp->pPreparedFwsecCmd);
            pKernelGsp->pPreparedFwsecCmd = NULL;
            return status;
        }
    }

    return NV_OK;
}

/*!
 * Obtain sysmem addr or arguments to be consumed by Booter Load.
 * Booter expects different arguments for normal boot, resume, and GC6 exit.
 *
 * @param[in]  bootMode  GSP boot mode
 */
static inline NvU64
_kgspGetBooterLoadArgs
(
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    switch (bootMode)
    {
        case KGSP_BOOT_MODE_NORMAL:
            return memdescGetPhysAddr(pKernelGsp->pWprMetaV1Descriptor, AT_GPU, 0);
        case KGSP_BOOT_MODE_SR_RESUME:
            return memdescGetPhysAddr(pKernelGsp->pSRMetaDescriptor, AT_GPU, 0);
        case KGSP_BOOT_MODE_GC6_EXIT:
        case KGSP_BOOT_MODE_SR_WITH_WPR_IN_SYSMEM:
            return 0;
    }

    // unreachable
    NV_ASSERT_FAILED("unexpected GSP boot mode");
    return 0;
}

/*
 * SEC2 can retain the first WPR-metadata DMA page across consecutive Booter
 * launches.  Merely moving the signature therefore does not make the second
 * launch consume the updated signature address.  Give every CMP50 launch a
 * distinct physical metadata page, then copy Booter's mutations back to the
 * canonical RM metadata before releasing the temporary page.
 */
static NV_STATUS
_kgspCmp50ExecuteBooterFreshMeta
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NV_STATUS status = NV_OK;
    NV_STATUS bootStatus = NV_OK;
    MEMORY_DESCRIPTOR *pFreshMeta = NULL;
    NvU8 *pFreshVa = NULL;
    NvU64 freshPhys;
    NvU64 flags = MEMDESC_FLAGS_ALLOC_IN_UNPROTECTED_MEMORY;

    if ((pKernelGsp->pWprMetaV1 == NULL) ||
        (pKernelGsp->pWprMetaV1Descriptor == NULL))
    {
        return NV_ERR_INVALID_STATE;
    }

    NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
        memdescCreate(&pFreshMeta, pGpu, 0x1000, 0x1000,
            NV_TRUE, ADDR_SYSMEM, NV_MEMORY_CACHED, flags), cleanup);
    memdescTagAlloc(status, NV_FB_ALLOC_RM_INTERNAL_OWNER_WPR_METADATA,
                    pFreshMeta);
    NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR, status, cleanup);

    pFreshVa = memdescMapInternal(pGpu, pFreshMeta, TRANSFER_FLAGS_NONE);
    NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
        (pFreshVa != NULL) ? NV_OK : NV_ERR_INSUFFICIENT_RESOURCES,
        cleanup);
    portMemSet(pFreshVa, 0, 0x1000);
    portMemCopy(pFreshVa, 0x1000, pKernelGsp->pWprMetaV1,
                sizeof(*pKernelGsp->pWprMetaV1));
    memdescUnmapInternal(pGpu, pFreshMeta, 0);
    pFreshVa = NULL;
    memdescFlushCpuCaches(pGpu, pFreshMeta);

    freshPhys = memdescGetPhysAddr(pFreshMeta, AT_GPU, 0);
    NV_PRINTF(LEVEL_ERROR,
              "CMP50_COMPUTE_UNLOCK_V140: fresh WPR metadata "
              "phys=0x%llx signature=0x%llx size=0x%llx\n",
              freshPhys,
              pKernelGsp->pWprMetaV1->sysmemAddrOfSignature,
              pKernelGsp->pWprMetaV1->sizeOfSignature);

    bootStatus = kgspExecuteBooterLoad_HAL(pGpu, pKernelGsp, freshPhys);

    pFreshVa = memdescMapInternal(pGpu, pFreshMeta, TRANSFER_FLAGS_NONE);
    NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
        (pFreshVa != NULL) ? NV_OK : NV_ERR_INSUFFICIENT_RESOURCES,
        cleanup);
    portMemCopy(pKernelGsp->pWprMetaV1,
                sizeof(*pKernelGsp->pWprMetaV1),
                pFreshVa, sizeof(*pKernelGsp->pWprMetaV1));
    memdescUnmapInternal(pGpu, pFreshMeta, 0);
    pFreshVa = NULL;
    memdescFlushCpuCaches(pGpu, pKernelGsp->pWprMetaV1Descriptor);
    status = bootStatus;

cleanup:
    if (pFreshVa != NULL)
    {
        memdescUnmapInternal(pGpu, pFreshMeta, 0);
    }
    if (pFreshMeta != NULL)
    {
        memdescFree(pFreshMeta);
        memdescDestroy(pFreshMeta);
    }
    return status;
}

/*!
 * Boot GSP-RM.
 *
 * This routine handles the following:
 *   - starts the RISCV core and passes control to boot binary image
 *   - waits for GSP-RM to complete initialization
 *
 * Note that boot binary and GSP-RM images have already been placed
 * in the appropriate places by kgspPopulateWprMeta_HAL().
 *
 * Note that this routine is based on flcnBootstrapRiscvOS_GA102().
 *
 * Note that this routine can be called without the API lock for
 * parllel initialization.
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      GSP object pointer
 * @param[in]   bootMode        GSP boot mode
 *
 * @return NV_OK if GSP-RM RISCV boot was successful.
 *         Appropriate NV_ERR_xxx value otherwise.
 */
NV_STATUS
kgspBootstrap_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    NV_STATUS status;
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);
    NvU32 cmp50StateIndex = s_cmp50StockflowGpuIndex(pGpu);
    NvBool bCmp50RetryRejoin = NV_FALSE;

    if ((bootMode == KGSP_BOOT_MODE_NORMAL) &&
        s_isCmp50ComputeUnlock(pGpu) &&
        g_cmp50V274PhaseCommitted[cmp50StateIndex])
    {
        NvU32 retryWprLo =
            GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
        NvU32 retryWprHi =
            GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
        NvBool bRetryWprDown =
            (retryWprHi == 0U) &&
            (retryWprLo == CMP50_WPR2_ADDR_LO_DOWN);
        NvBool bRetryWprRange =
            (retryWprLo != 0U) && (retryWprLo < retryWprHi);

        if (!(bRetryWprDown || bRetryWprRange) ||
            (GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM) !=
             CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
            (GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0) !=
             CMP50_FECS_SM_SPEED_FULL_0) ||
            (GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1) !=
             CMP50_FECS_SM_SPEED_FULL_1))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_STOCKFLOW_V551: "
                      "RETRY_REJOIN_STATE_MISMATCH idx=%u "
                      "WPR=%08x:%08x FECS=%08x SS0=%08x SS1=%08x\n",
                      cmp50StateIndex, retryWprHi, retryWprLo,
                      GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM),
                      GPU_REG_RD32(pGpu,
                          CMP50_FECS_SM_SPEED_OVERRIDE_0),
                      GPU_REG_RD32(pGpu,
                          CMP50_FECS_SM_SPEED_OVERRIDE_1));
            return NV_ERR_INVALID_STATE;
        }

        if (g_cmp50V274GspReady[cmp50StateIndex])
        {
            NvU32 plm;
            NvU32 speed0;
            NvU32 speed1;

            if (pKernelGsp->pPreparedFwsecCmd == NULL)
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP50_STOCKFLOW_V551: "
                          "PERSISTENT_READY_NO_FWSEC idx=%u "
                          "WPR=%08x:%08x\n",
                          cmp50StateIndex, retryWprHi, retryWprLo);
                return NV_ERR_INVALID_STATE;
            }

            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_STOCKFLOW_V551: "
                      "PERSISTENT_READY_REFWSEC idx=%u "
                      "WPR=%08x:%08x\n",
                      cmp50StateIndex, retryWprHi, retryWprLo);
            NV_ASSERT_OK_OR_RETURN(kflcnReset_HAL(pGpu, pKernelFalcon));
            status = kgspExecuteFwsec_HAL(
                pGpu, pKernelGsp, pKernelGsp->pPreparedFwsecCmd);
            portMemFree(pKernelGsp->pPreparedFwsecCmd);
            pKernelGsp->pPreparedFwsecCmd = NULL;
            if (status != NV_OK)
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP50_STOCKFLOW_V551: "
                          "PERSISTENT_READY_REFWSEC_FAIL status=0x%x "
                          "idx=%u\n",
                          status, cmp50StateIndex);
                return status;
            }

            retryWprLo =
                GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
            retryWprHi =
                GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
            plm = GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM);
            speed0 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0);
            speed1 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1);
            if ((g_cmp50StockWprLo[cmp50StateIndex] == 0U) ||
                (g_cmp50StockWprHi[cmp50StateIndex] == 0U) ||
                (retryWprLo != g_cmp50StockWprLo[cmp50StateIndex]) ||
                (retryWprHi != g_cmp50StockWprHi[cmp50StateIndex]) ||
                (plm != CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
                (speed0 != CMP50_FECS_SM_SPEED_FULL_0) ||
                (speed1 != CMP50_FECS_SM_SPEED_FULL_1))
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP50_STOCKFLOW_V551: "
                          "PERSISTENT_READY_REFWSEC_STATE_MISMATCH "
                          "idx=%u WPR=%08x:%08x FECS=%08x "
                          "SS0=%08x SS1=%08x\n",
                          cmp50StateIndex, retryWprHi, retryWprLo,
                          plm, speed0, speed1);
                return NV_ERR_INVALID_STATE;
            }
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_STOCKFLOW_V551: "
                      "PERSISTENT_READY_REFWSEC_STOCK_WPR_PASS "
                      "idx=%u WPR=%08x:%08x FECS=%08x "
                      "SS0=%08x SS1=%08x\n",
                      cmp50StateIndex, retryWprHi, retryWprLo,
                      plm, speed0, speed1);

            status = kgspCmp50RebuildStockSignature(pGpu, pKernelGsp);
            if (status != NV_OK)
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP50_STOCKFLOW_V551: "
                          "PERSISTENT_READY_STOCK_SIGNATURE_FAIL "
                          "status=0x%x idx=%u\n",
                          status, cmp50StateIndex);
                return status;
            }
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_STOCKFLOW_V551: "
                      "PERSISTENT_READY_STOCK_SIGNATURE_PASS idx=%u\n",
                      cmp50StateIndex);
        }
        kgspCmp50SetExploitMode(pGpu, NV_FALSE);
        NV_ASSERT_OK_OR_RETURN(kflcnResetIntoRiscv_HAL(pGpu, pKernelFalcon));
        kgspProgramLibosBootArgsAddr_HAL(pGpu, pKernelGsp);
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_STOCKFLOW_V551: RETRY_REJOIN_STOCK_BOOTER "
                  "idx=%u WPR=%08x:%08x\n",
                  cmp50StateIndex, GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI), GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO));
        bCmp50RetryRejoin = NV_TRUE;
        goto cmp50_v525_stock_booter;
    }

cmp50_v274_commit:
    // V274 commits after FWSEC/FRTS but before switching GSP into RISC-V mode.
    if ((bootMode == KGSP_BOOT_MODE_NORMAL) &&
        s_isCmp50ComputeUnlock(pGpu) &&
        g_cmp50V274FwsecReady[cmp50StateIndex])
    {
        NvU32 wpr2Lo = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
        NvU32 wpr2Hi = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
        NvU32 plm;
        NvU32 handoffWprLo;
        NvU32 handoffWprHi;
        NvU32 speed0;
        NvU32 speed1;

        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V274: POST_FWSEC_PRE_GSP_ENTRY "
                  "WPR=%08x:%08x FECS=%08x RESET=%08x\n",
                  wpr2Hi, wpr2Lo,
                  GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM),
                  GPU_REG_RD32(pGpu, CMP50_SEC2_RESET_PLM));

        // At this point a cold function normally has WPR down.  Also accept
        // the valid range form so the test remains bounded across RM retries.
        if (!((wpr2Hi == 0U) ||
              ((wpr2Lo != 0U) && (wpr2Lo < wpr2Hi))))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V274: unexpected post-FWSEC WPR "
                      "%08x:%08x\n", wpr2Hi, wpr2Lo);
            return NV_ERR_INVALID_STATE;
        }

        /*
         * The stock WPR lives at the top of framebuffer memory.  The 10 GiB
         * board reports 027fee00:027fe000, while the 20 GiB board reports
         * 04ffee00:04ffe000.  Preserve the range produced by this card's
         * successful FWSEC run instead of comparing against one SKU's
         * hard-coded addresses.  Only accept the known stock span and never
         * learn a value from the exploit's WPR-down sentinel state.
         */
        if ((wpr2Lo != 0U) && (wpr2Hi > wpr2Lo))
        {
            if ((wpr2Hi - wpr2Lo) != CMP50_WPR2_STOCK_SPAN)
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP50_WPR_DYNAMIC_V1: STOCK_SPAN_MISMATCH "
                          "idx=%u WPR=%08x:%08x span=%08x\n",
                          cmp50StateIndex, wpr2Hi, wpr2Lo,
                          wpr2Hi - wpr2Lo);
                return NV_ERR_INVALID_STATE;
            }
            g_cmp50StockWprLo[cmp50StateIndex] = wpr2Lo;
            g_cmp50StockWprHi[cmp50StateIndex] = wpr2Hi;
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_WPR_DYNAMIC_V1: STOCK_RANGE_CAPTURED "
                      "idx=%u WPR=%08x:%08x span=%08x\n",
                      cmp50StateIndex, wpr2Hi, wpr2Lo,
                      wpr2Hi - wpr2Lo);
        }
        else if ((g_cmp50StockWprLo[cmp50StateIndex] == 0U) ||
                 (g_cmp50StockWprHi[cmp50StateIndex] == 0U))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_WPR_DYNAMIC_V1: STOCK_RANGE_UNAVAILABLE "
                      "idx=%u WPR=%08x:%08x\n",
                      cmp50StateIndex, wpr2Hi, wpr2Lo);
            return NV_ERR_INVALID_STATE;
        }

        kgspCmp50SetExploitMode(pGpu, NV_TRUE);
        status = _kgspCmp50ExecuteBooterFreshMeta(pGpu, pKernelGsp);
        kgspCmp50SetExploitMode(pGpu, NV_FALSE);
        plm = GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM);
        handoffWprLo = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
        handoffWprHi = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
        speed0 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0);
        speed1 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1);
        if ((status != NV_OK) ||
            (plm != CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
            (handoffWprLo != CMP50_WPR2_ADDR_LO_DOWN) ||
            (handoffWprHi != 0U) ||
            (speed0 != CMP50_FECS_SM_SPEED_FULL_0) ||
            (speed1 != CMP50_FECS_SM_SPEED_FULL_1))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V274: POST_FWSEC_COMMIT_FAIL "
                      "status=%x WPR=%08x:%08x FECS=%08x "
                      "SS0=%08x SS1=%08x\n",
                      status, handoffWprHi, handoffWprLo, plm,
                      speed0, speed1);
            return NV_ERR_INVALID_STATE;
        }

        status = kgspCmp50RebuildStockSignature(pGpu, pKernelGsp);
        if (status != NV_OK)
            return status;

        /*
         * The signed verifier tail marks this metadata verified and advances
         * bootCount before the protected cleanup drops WPR2. Reusing those
         * output flags makes the next stock Booter return NV_OK without
         * rebuilding WPR2 or starting GSP-RM. Restore only the two cold-input
         * fields initialized by kgspPopulateWprMeta_TU102; keep every layout
         * address and firmware pointer produced by the stock path.
         */
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V523: WPR_META_BEFORE_RESET "
                  "bootCount=0x%016llx verified=0x%016llx\n",
                  pKernelGsp->pWprMetaV1->bootCount,
                  pKernelGsp->pWprMetaV1->verified);
        pKernelGsp->pWprMetaV1->bootCount = 0;
        pKernelGsp->pWprMetaV1->verified = 0;
        memdescFlushCpuCaches(pGpu, pKernelGsp->pWprMetaV1Descriptor);
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V523: "
                  "WPR_META_INPUT_RESET_PASS bootCount=0 verified=0\n");

        status = kgspCmp50SanitizeSec2AfterExploit(pGpu, pKernelGsp);
        if (status != NV_OK)
            return status;

        plm = GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM);
        handoffWprLo = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
        handoffWprHi = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
        speed0 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0);
        speed1 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1);
        if ((plm != CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
            (handoffWprLo != CMP50_WPR2_ADDR_LO_DOWN) ||
            (handoffWprHi != 0U) ||
            (speed0 != CMP50_FECS_SM_SPEED_FULL_0) ||
            (speed1 != CMP50_FECS_SM_SPEED_FULL_1))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V276: SANITIZE_STATE_LOST "
                      "WPR=%08x:%08x FECS=%08x SS0=%08x SS1=%08x\n",
                      handoffWprHi, handoffWprLo, plm, speed0, speed1);
            return NV_ERR_INVALID_STATE;
        }

        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V526: "
                  "FULLSPEED_STOCK_RELOCK_WPR_DOWN_HANDOFF_PASS "
                  "WPR=%08x:%08x FECS=%08x RESET=%08x "
                  "SS0=%08x SS1=%08x\n",
                  handoffWprHi, handoffWprLo, plm,
                  GPU_REG_RD32(pGpu, CMP50_SEC2_RESET_PLM),
                  speed0, speed1);

        /*
         * V551 single-module stock-flow experiment. V534 stopped here so an
         * external transaction could unload this module and hand the preserved
         * WPR-down/full-speed state to a separately hashed V529 stock path.
         * For the patched-driver lane, keep the same exact handoff gate, then
         * continue into the already staged V524/V525 stock FWSEC/RISCV/Booter
         * continuation below.
         */
        if (GPU_REG_RD32(pGpu, CMP50_SEC2_RESET_PLM) != 0x000000FFU)
            return NV_ERR_INVALID_STATE;
        g_cmp50V274PhaseCommitted[cmp50StateIndex] = NV_TRUE;
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V534: "
                  "TOKEN_RELEASE_RESETPLM_FF_PHASE_BOUNDARY_PASS "
                  "idx=%u WPR=%08x:%08x FECS=%08x RESET=%08x "
                  "SS0=%08x SS1=%08x retry_guard=armed\n",
                  cmp50StateIndex,
                  handoffWprHi, handoffWprLo, plm,
                  GPU_REG_RD32(pGpu, CMP50_SEC2_RESET_PLM),
                  speed0, speed1);
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_STOCKFLOW_V551: BDF_STATE "
                  "idx=%u bus=%u device=%u gpuInstance=%u "
                  "from_v534_phase_boundary\n",
                  cmp50StateIndex, gpuGetBus(pGpu), gpuGetDevice(pGpu),
                  pGpu->gpuInstance);

        /*
         * Stock TU102 establishes the FRTS/WPR2 range with FWSEC before
         * Booter Load. V522/V523 skipped directly from signed WPR-down
         * cleanup to Booter, which returned NV_OK without raising WPR2.
         * Replay the untouched driver-prepared FWSEC command and require the
         * exact stock range while retaining the protected full-speed state.
         */
        if (pKernelGsp->pPreparedFwsecCmd == NULL)
            return NV_ERR_INVALID_STATE;
        NV_ASSERT_OK_OR_RETURN(kflcnReset_HAL(pGpu, pKernelFalcon));
        status = kgspExecuteFwsec_HAL(
            pGpu, pKernelGsp, pKernelGsp->pPreparedFwsecCmd);
        portMemFree(pKernelGsp->pPreparedFwsecCmd);
        pKernelGsp->pPreparedFwsecCmd = NULL;
        if (status != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V524: REFWSEC_FAIL status=0x%x\n",
                      status);
            return status;
        }

        plm = GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM);
        handoffWprLo = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
        handoffWprHi = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
        speed0 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0);
        speed1 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1);
        if ((handoffWprLo != g_cmp50StockWprLo[cmp50StateIndex]) ||
            (handoffWprHi != g_cmp50StockWprHi[cmp50StateIndex]) ||
            (plm != CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
            (speed0 != CMP50_FECS_SM_SPEED_FULL_0) ||
            (speed1 != CMP50_FECS_SM_SPEED_FULL_1))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V524: REFWSEC_STATE_MISMATCH "
                      "status=0x%x WPR=%08x:%08x FECS=%08x "
                      "SS0=%08x SS1=%08x\n",
                      status, handoffWprHi, handoffWprLo, plm,
                      speed0, speed1);
            return NV_ERR_INVALID_STATE;
        }
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V524: REFWSEC_STOCK_WPR_PASS "
                  "WPR=%08x:%08x FECS=%08x SS0=%08x SS1=%08x\n",
                  handoffWprHi, handoffWprLo, plm, speed0, speed1);

        /*
         * Rejoin the two stock instructions that normally run immediately
         * after FWSEC and before Booter Load. V524 jumped over both when it
         * entered the in-place Booter label, leaving GSP in Falcon mode with
         * no LibOS boot-argument address even though FWSEC and Booter passed.
         */
        NV_ASSERT_OK_OR_RETURN(
            kflcnResetIntoRiscv_HAL(pGpu, pKernelFalcon));
        kgspProgramLibosBootArgsAddr_HAL(pGpu, pKernelGsp);
        if ((GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO) !=
             g_cmp50StockWprLo[cmp50StateIndex]) ||
            (GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI) !=
             g_cmp50StockWprHi[cmp50StateIndex]) ||
            (GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM) !=
             CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
            (GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0) !=
             CMP50_FECS_SM_SPEED_FULL_0) ||
            (GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1) !=
             CMP50_FECS_SM_SPEED_FULL_1))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V525: "
                      "RISCV_BOOTARGS_STATE_MISMATCH\n");
            return NV_ERR_INVALID_STATE;
        }
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V525: "
                  "RISCV_MODE_BOOTARGS_PASS WPR=%08x:%08x "
                  "FECS=%08x SS0=%08x SS1=%08x\n",
                  GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI),
                  GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO),
                  GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM),
                  GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0),
                  GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1));

        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V274: POST_FWSEC_PRE_GSP_HANDOFF_PASS "
                  "WPR=%08x:%08x FECS=%08x RESET=%08x "
                  "SS0=%08x SS1=%08x\n",
                  handoffWprHi, handoffWprLo, plm,
                  GPU_REG_RD32(pGpu, CMP50_SEC2_RESET_PLM),
                  speed0, speed1);
        g_cmp50V274PhaseCommitted[cmp50StateIndex] = NV_TRUE;
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V525: INPLACE_STOCK_BOOTER_COMMIT "
                  "idx=%u retry_guard=armed\n",
                  cmp50StateIndex);
        goto cmp50_v525_stock_booter;
    }

    // Execute Scrubber if needed
    if (((bootMode == KGSP_BOOT_MODE_SR_RESUME) || (bootMode == KGSP_BOOT_MODE_NORMAL)) &&
        (pKernelGsp->pScrubberUcode != NULL))
    {
        NV_ASSERT_OK_OR_RETURN(kgspExecuteScrubberIfNeeded_HAL(pGpu, pKernelGsp));
    }

    //
    // For normal boot, additional setup is necessary.
    // Note: for resume or GC6 exit, Booter and/or GSP-RM will restore these.
    //
    if (bootMode == KGSP_BOOT_MODE_NORMAL)
    {
        // Execute FWSEC to setup FRTS if we have a FRTS region.
        if (kgspGetFrtsSize_HAL(pGpu, pKernelGsp) > 0)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "SEC2_DEBUG: FWSEC: pPreparedFwsecCmd=%p frtsSize=0x%x\n",
                      pKernelGsp->pPreparedFwsecCmd,
                      kgspGetFrtsSize_HAL(pGpu, pKernelGsp));

            if (pKernelGsp->pPreparedFwsecCmd == NULL)
            {
                NV_PRINTF(LEVEL_ERROR, "SEC2_DEBUG: FWSEC cmd is NULL, aborting\n");
                return NV_ERR_INVALID_STATE;
            }

            status = kflcnReset_HAL(pGpu, pKernelFalcon);
            NV_PRINTF(LEVEL_ERROR,
                      "SEC2_DEBUG: kflcnReset for FWSEC: 0x%x\n", status);
            if (status != NV_OK) return status;

            status = kgspExecuteFwsec_HAL(pGpu, pKernelGsp, pKernelGsp->pPreparedFwsecCmd);
            /*
             * Retain the stock prepared FRTS command on the exact CMP50
             * target for one bounded replay after the signed WPR-down
             * transaction. Other devices keep the stock lifetime.
             */
            if (!s_isCmp50ComputeUnlock(pGpu))
            {
                portMemFree(pKernelGsp->pPreparedFwsecCmd);
                pKernelGsp->pPreparedFwsecCmd = NULL;
            }

            NV_PRINTF(LEVEL_ERROR,
                      "SEC2_DEBUG: FWSEC status=0x%x\n", status);
            if (status != NV_OK) return status;
        }

        if (s_isCmp50ComputeUnlock(pGpu))
        {
            g_cmp50V274FwsecReady[cmp50StateIndex] = NV_TRUE;
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V274: "
                      "FWSEC_COMPLETE_GSP_UNTOUCHED idx=%u\n",
                      cmp50StateIndex);
            goto cmp50_v274_commit;
        }

        status = kflcnResetIntoRiscv_HAL(pGpu, pKernelFalcon);
        NV_PRINTF(LEVEL_ERROR,
                  "SEC2_DEBUG: kflcnResetIntoRiscv: 0x%x\n", status);
        if (status != NV_OK) return status;

        // Load init args into mailbox regs
        kgspProgramLibosBootArgsAddr_HAL(pGpu, pKernelGsp);
    }

cmp50_v525_stock_booter:
    // Execute stock Booter Load in the same RM/WPR-metadata context.
    if ((bootMode == KGSP_BOOT_MODE_NORMAL) &&
        s_isCmp50ComputeUnlock(pGpu) &&
        !bCmp50RetryRejoin)
    {
        status = _kgspCmp50ExecuteBooterFreshMeta(pGpu, pKernelGsp);
    }
    else
    {
        status = kgspExecuteBooterLoad_HAL(
            pGpu, pKernelGsp,
            _kgspGetBooterLoadArgs(pKernelGsp, bootMode));
    }

    {
        NvU32 devId = pGpu->idInfo.PCIDeviceID >> 16;
        if (devId == 0x20C2 || devId == 0x2082)
            NV_PRINTF(LEVEL_ERROR,
                      "SEC2_DEBUG: normal BooterLoad status=0x%x\n", status);
    }

    if ((bootMode == KGSP_BOOT_MODE_NORMAL) &&
        (status == NV_OK) &&
        s_isCmp50ComputeUnlock(pGpu))
    {
        NvU32 postWprLo = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
        NvU32 postWprHi = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
        NvU32 postSpeed0 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_0);
        NvU32 postSpeed1 = GPU_REG_RD32(pGpu, CMP50_FECS_SM_SPEED_OVERRIDE_1);
        NvU32 postPlm = GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_PLM);
        NvU32 postResetPlm =
            GPU_REG_RD32(pGpu, CMP50_SEC2_RESET_PLM);
        if ((postWprLo >= postWprHi) ||
            (postSpeed0 != CMP50_FECS_SM_SPEED_FULL_0) ||
            (postSpeed1 != CMP50_FECS_SM_SPEED_FULL_1) ||
            (postPlm != CMP50_FECS_FEATURE_PLM_POSTLOCK) ||
            ((postResetPlm != CMP50_SEC2_RESET_PLM_STOCK) &&
             (postResetPlm != 0x000000FFU)))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_COMPUTE_UNLOCK_V525: stock Booter mismatch "
                      "WPR=%08x:%08x SS0=0x%08x SS1=0x%08x "
                      "FECS=0x%08x RESET=0x%08x\n",
                      postWprHi, postWprLo, postSpeed0, postSpeed1,
                      postPlm, postResetPlm);
            return NV_ERR_INVALID_STATE;
        }
        if (postResetPlm == 0x000000FFU)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_STOCKFLOW_V551: "
                      "STOCK_BOOTER_RESET_FF_ACCEPTED\n");
        }
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_COMPUTE_UNLOCK_V525: INPLACE_STOCK_BOOTER_AFTER_UNLOCK_PASS "
                  "WPR=%08x:%08x SS0=0x%08x SS1=0x%08x "
                  "FECS_PLM=0x%08x RESET_PLM=0x%08x\n",
                  postWprHi, postWprLo, postSpeed0, postSpeed1,
                  postPlm, postResetPlm);
        s_cmp50ApplyGen2Policy(pGpu, "post-booter");
    }

    {
        NvU32 devId = pGpu->idInfo.PCIDeviceID >> 16;
        if ((devId == 0x20C2 || devId == 0x2082) && status == NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "SEC2_DEBUG: POST-BooterLoad verify PLM=0x%08x SS0=0x%08x SS1=0x%08x "
                      "CFG1=0x%08x LMR=0x%08x\n",
                      GPU_REG_RD32(pGpu, SEC2_DEBUG_PRI_FEATURE_OVERRIDE_PLM),
                      GPU_REG_RD32(pGpu, SEC2_DEBUG_PRI_FEATURE_OVERRIDE_SM_SPEED),
                      GPU_REG_RD32(pGpu, SEC2_DEBUG_PRI_FEATURE_OVERRIDE_SM_SPEED_1),
                      GPU_REG_RD32(pGpu, SEC2_DEBUG_PRI_FBPA_CFG1),
                      GPU_REG_RD32(pGpu, SEC2_DEBUG_PRI_MMU_LMR));
        }
    }


    {
        NvU32 lateDevId = pGpu->idInfo.PCIDeviceID >> 16;
        if ((lateDevId == 0x20C2 || lateDevId == 0x2082) && status == NV_OK)
        {
            #define PCIE_GEN2_PRIV_MISC_1_ADDR_LATE     0x0008841cU
            #define PCIE_GEN2_PRIV_MISC_1_GEN2_EN_LATE  ((1U << 11) | (1U << 13))
            #define PCIE_GEN2_PRIV_MISC_1_GEN2_VAL_LATE ((1U << 12) | (1U << 14))
            NvU32 misc1Late = GPU_REG_RD32(pGpu, PCIE_GEN2_PRIV_MISC_1_ADDR_LATE);
            NvU32 misc1WantLate = (misc1Late | PCIE_GEN2_PRIV_MISC_1_GEN2_EN_LATE)
                                  & ~PCIE_GEN2_PRIV_MISC_1_GEN2_VAL_LATE;
            GPU_REG_WR32(pGpu, PCIE_GEN2_PRIV_MISC_1_ADDR_LATE, misc1WantLate);
            NV_PRINTF(LEVEL_ERROR,
                      "SEC2_DEBUG: PCIe PRIV_MISC_1 late pre=0x%08x post=0x%08x\n",
                      misc1Late,
                      GPU_REG_RD32(pGpu, PCIE_GEN2_PRIV_MISC_1_ADDR_LATE));
            {
                NvU32 cyaLate = GPU_REG_RD32(pGpu, 0x0008c2c0U);
                cyaLate = cyaLate & ~(1U << 2);
                GPU_REG_WR32(pGpu, 0x0008c2c0U, cyaLate);
                NV_PRINTF(LEVEL_ERROR,
                          "SEC2_DEBUG: PCIe CYA_0 late clear DIS_G2: 0x%08x (bit2=%u)\n",
                          GPU_REG_RD32(pGpu, 0x0008c2c0U),
                          (GPU_REG_RD32(pGpu, 0x0008c2c0U) >> 2) & 1U);
                NvU32 linkCfgLate = GPU_REG_RD32(pGpu, 0x0008c040U);
                linkCfgLate = (linkCfgLate & ~0x000C0000U) | (0x2U << 18);
                GPU_REG_WR32(pGpu, 0x0008c040U, linkCfgLate);
                NV_PRINTF(LEVEL_ERROR,
                          "SEC2_DEBUG: PCIe LINK_CONFIG_0 late MAX_RATE=2: 0x%08x\n",
                          GPU_REG_RD32(pGpu, 0x0008c040U));
                GPU_REG_WR32(pGpu, 0x0008872cU, 0x00000006U);
                NV_PRINTF(LEVEL_ERROR,
                          "SEC2_DEBUG: PCIe XVE_OVR late=0x%08x\n",
                          GPU_REG_RD32(pGpu, 0x0008872cU));
            }
        }
    }

    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "failed to execute Booter Load (ucode for initial boot): 0x%x\n", status);
        return status;
    }

    // Program FALCON_OS
    RM_RISCV_UCODE_DESC *pRiscvDesc = pKernelGsp->pGspRmBootUcodeDesc;
    kflcnRegWrite_HAL(pGpu, pKernelFalcon, NV_PFALCON_FALCON_OS, pRiscvDesc->appVersion);

    // Ensure the CPU has started
    // Note: In rare cases, GSP-RM may make enough progress by this point to suspend waiting for Kernel RM.
    if (kflcnIsRiscvActive_HAL(pGpu, pKernelFalcon) || _kgspIsProcessorSuspended(pGpu, pKernelGsp))
    {
        NV_PRINTF(LEVEL_INFO, "GSP ucode loaded and RISCV started.\n");
    }
    else
    {
        NV_PRINTF(LEVEL_ERROR, "Failed to boot GSP.\n");

        return NV_ERR_NOT_READY;
    }

    NV_PRINTF(LEVEL_INFO, "Waiting for GSP fw RM to be ready...\n");

    if (bootMode == KGSP_BOOT_MODE_NORMAL)
    {
        //
        // For normal boot, link the status queue.
        // Note: for resume or GC6 exit, GSP-RM will restore queue state.
        //
        NV_ASSERT_OK_OR_RETURN(GspStatusQueueInit(pGpu, &pKernelGsp->pRpc->pMessageQueueInfo));
        //
        // Send GSP_INIT: system info in, GSP static info out.
        //
        NV_ASSERT_OK_OR_RETURN(kgspSendInitRpcs(pGpu, pKernelGsp));
    }
    else
    {
        NV_ASSERT_OK_OR_RETURN(kgspWaitForRmResumeDone(pGpu, pKernelGsp));
    }

    NV_PRINTF(LEVEL_INFO, "GSP FW RM ready.\n");

    if ((pGpu->idInfo.PCIDeviceID == 0x1E0910DEU) &&
        ((pGpu->idInfo.PCISubDeviceID == 0x155410DEU) || (pGpu->idInfo.PCISubDeviceID == 0x371F1462U)))
    {
        s_cmp50ApplyGen2Policy(pGpu, "gsp-ready");
        g_cmp50V274GspReady[cmp50StateIndex] = NV_TRUE;
        NV_PRINTF(LEVEL_ERROR,
                  "CMP50_GSP_READY_V551: feature=0x%08x ss0=0x%08x "
                  "readout=0x%08x ss1=0x%08x\n",
                  GPU_REG_RD32(pGpu, CMP50_FECS_FEATURE_READOUT),
                  GPU_REG_RD32(pGpu,
                      CMP50_FECS_FEATURE_OVERRIDE_SM_SPEED_SELECT),
                  GPU_REG_RD32(pGpu,
                      CMP50_FECS_FEATURE_READOUT_SM_SPEED_SELECT),
                  GPU_REG_RD32(pGpu,
                      CMP50_FECS_FEATURE_OVERRIDE_SM_SPEED_SELECT_1));
    }

    return NV_OK;
}

/*!
 * Obtain sysmem addr or arguments to be consumed by Booter Unload.
 * Booter expects different arguments for normal unload, suspend, and GC6 enter.
 *
 * @param[in]  unloadMode  GSP unload mode
 */
static inline NvU64
_kgspGetBooterUnloadArgs
(
    KernelGsp *pKernelGsp,
    KernelGspUnloadMode unloadMode
)
{
    switch (unloadMode)
    {
        case KGSP_UNLOAD_MODE_NORMAL:
        case KGSP_UNLOAD_MODE_GC6_ENTER:
        case KGSP_UNLOAD_MODE_SR_WITH_WPR_IN_SYSMEM:
            return 0;
        case KGSP_UNLOAD_MODE_SR_SUSPEND:
            return memdescGetPhysAddr(pKernelGsp->pSRMetaDescriptor, AT_GPU, 0);
    }

    // unreachable
    NV_ASSERT_FAILED("unexpected GSP unload mode");
    return 0;
}

/*!
 * Teardown remaining GSP state after GSP-RM unloads.
 *
 * For pre-Hopper, this involves running FWSEC-SB to put back pre-OS apps and
 * Booter Unload to teardown WPR2.
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      GSP object pointer
 * @param[in]   unloadMode      GSP unload mode
 */
NV_STATUS
kgspTeardown_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspUnloadMode unloadMode
)
{
    NV_STATUS status;

    //
    // Avoid cascading timeouts when attempting to invoke the below ucodes if
    // we are unloading due to a GSP-RM timeout.
    //
    threadStateResetTimeout(pGpu);

    if (unloadMode != KGSP_UNLOAD_MODE_GC6_ENTER)
    {
        KernelGspPreparedFwsecCmd preparedCmd;

        // Reset GSP so we can load FWSEC-SB
        status = kflcnReset_HAL(pGpu, staticCast(pKernelGsp, KernelFalcon));
        NV_ASSERT((status == NV_OK) || (status == NV_ERR_GPU_IN_FULLCHIP_RESET));

        // Invoke FWSEC-SB to put back PreOsApps during driver unload
        status = kgspPrepareForFwsecSb_HAL(pGpu, pKernelGsp, pKernelGsp->pFwsecUcode, &preparedCmd);
        if (status != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR, "failed to prepare for FWSEC-SB for PreOsApps during driver unload: 0x%x\n", status);
            NV_ASSERT_FAILED("FWSEC-SB prep failed");
        }
        else
        {
            status = kgspExecuteFwsec_HAL(pGpu, pKernelGsp, &preparedCmd);
            if ((status != NV_OK) && (status != NV_ERR_GPU_IN_FULLCHIP_RESET))
            {
                NV_PRINTF(LEVEL_ERROR, "failed to execute FWSEC-SB for PreOsApps during driver unload: 0x%x\n", status);
                NV_ASSERT_FAILED("FWSEC-SB failed");
            }
        }
    }

    // Execute Booter Unload
    status = kgspExecuteBooterUnloadIfNeeded_HAL(pGpu, pKernelGsp,
                                                 _kgspGetBooterUnloadArgs(pKernelGsp, unloadMode));
    return status;
}

void
kgspGetGspRmBootUcodeStorage_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    BINDATA_STORAGE **ppBinStorageImage,
    BINDATA_STORAGE **ppBinStorageDesc
)
{
    const BINDATA_ARCHIVE *pBinArchive = kgspGetBinArchiveGspRmBoot_HAL(pKernelGsp);

    if (pBinArchive == NULL)
    {
        if (s_isCmp50ComputeUnlock(pGpu))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "CMP50_STOCKFLOW_V551: GSP_RM_BOOT_ARCHIVE_NULL\n");
        }
        *ppBinStorageImage = NULL;
        *ppBinStorageDesc = NULL;
        return;
    }

    *ppBinStorageImage = (BINDATA_STORAGE *) bindataArchiveGetStorage(pBinArchive, BINDATA_LABEL_UCODE_IMAGE);
    *ppBinStorageDesc  = (BINDATA_STORAGE *) bindataArchiveGetStorage(pBinArchive, BINDATA_LABEL_UCODE_DESC);
}

/*!
 * Populate WPR meta structure.
 *
 * Firmware scrubs the last 256mb of FB, no memory outside of this region
 * may be used until the FW RM has scrubbed the remainder of memory.
 *
 *   ---------------------------- <- fbSize (end of FB, 1M aligned)
 *   | VGA WORKSPACE            |
 *   ---------------------------- <- vbiosReservedOffset  (64K? aligned)
 *   | (potential align. gap)   |
 *   ---------------------------- <- gspFwWprEnd (128K aligned)
 *   | FRTS data                |    (frtsSize is 0 on GA100)
 *   | ------------------------ | <- frtsOffset
 *   | BOOT BIN (e.g. SK + BL)  |
 *   ---------------------------- <- bootBinOffset
 *   | GSP FW ELF               |
 *   ---------------------------- <- gspFwOffset
 *   | GSP FW (WPR) HEAP        |
 *   ---------------------------- <- gspFwHeapOffset**
 *   | Booter-placed metadata   |
 *   | (struct GspFwWprMeta)    |
 *   ---------------------------- <- gspFwWprStart (128K aligned)
 *   | GSP FW (non-WPR) HEAP    |
 *   ---------------------------- <- nonWprHeapOffset, gspFwRsvdStart
 *
 *  gspFwHeapOffset** contains the entire WPR heap region, which can be subdivided
 *  for various GSP FW components.
 *
 * @param       pGpu          GPU object pointer
 * @param       pKernelGsp    KernelGsp object pointer
 * @param       pGspFw        Pointer to GSP-RM fw image.
 */
NV_STATUS
kgspPopulateWprMeta_TU102
(
    OBJGPU         *pGpu,
    KernelGsp      *pKernelGsp,
    GSP_FIRMWARE   *pGspFw
)
{
    KernelMemorySystem  *pKernelMemorySystem  = GPU_GET_KERNEL_MEMORY_SYSTEM(pGpu);
    KernelDisplay       *pKernelDisplay = GPU_GET_KERNEL_DISPLAY(pGpu);
    MemoryManager       *pMemoryManager = GPU_GET_MEMORY_MANAGER(pGpu);
    GspFwWprMetaV1      *pWprMeta = pKernelGsp->pWprMetaV1;
    RM_RISCV_UCODE_DESC *pRiscvDesc = pKernelGsp->pGspRmBootUcodeDesc;
    NvU64                vbiosReservedOffset;
    NvU64                mmuLockLo, mmuLockHi;
    NvBool               bIsMmuLockValid;
    NvU32                data;

    ct_assert(sizeof(*pWprMeta) == 256);

    NV_ASSERT_OR_RETURN(IS_GSP_CLIENT(pGpu), NV_ERR_NOT_SUPPORTED);

    NV_ASSERT_OR_RETURN(pKernelGsp->pGspRmBootUcodeImage != NULL, NV_ERR_INVALID_STATE);
    NV_ASSERT_OR_RETURN(pKernelGsp->gspRmBootUcodeSize != 0, NV_ERR_INVALID_STATE);
    NV_ASSERT_OR_RETURN(pRiscvDesc != NULL, NV_ERR_INVALID_STATE);

    NV_ASSERT_OK_OR_RETURN(kmemsysGetUsableFbSize_HAL(pGpu, pKernelMemorySystem, &pWprMeta->fbSize));

    //
    // Start layout calculations at the top and work down.
    // Figure out where VGA workspace is located.  We do not have to adjust
    // it ourselves (see vgaRelocateWorkspaceBase_HAL()).
    //
    if (gpuFuseSupportsDisplay_HAL(pGpu) &&
        kdispGetVgaWorkspaceBase(pGpu, pKernelDisplay, &pWprMeta->vgaWorkspaceOffset))
    {
        if (pWprMeta->vgaWorkspaceOffset < (pWprMeta->fbSize - DRF_SIZE(NV_PRAMIN)))
        {
            const NvU32 VBIOS_WORKSPACE_SIZE = 0x20000;

            // Point NV_PDISP_VGA_WORKSPACE_BASE to end-of-FB
            pWprMeta->vgaWorkspaceOffset = (pWprMeta->fbSize - VBIOS_WORKSPACE_SIZE);
        }
    }
    else
    {
        pWprMeta->vgaWorkspaceOffset = (pWprMeta->fbSize - DRF_SIZE(NV_PRAMIN));
    }
    pWprMeta->vgaWorkspaceSize = pWprMeta->fbSize - pWprMeta->vgaWorkspaceOffset;

    // Check for MMU locked region (locked by VBIOS)
    NV_ASSERT_OK_OR_RETURN(
        memmgrReadMmuLock_HAL(pGpu, pMemoryManager, &bIsMmuLockValid, &mmuLockLo, &mmuLockHi));

    if (bIsMmuLockValid)
        vbiosReservedOffset = NV_MIN(mmuLockLo, pWprMeta->vgaWorkspaceOffset);
    else
        vbiosReservedOffset = pWprMeta->vgaWorkspaceOffset;

    // Set the size of the GSP FW ahead of kgspGetWprEndMargin()
    pWprMeta->sizeOfRadix3Elf = pGspFw->imageSize;

    // End of WPR region (128KB aligned), shifted for any WPR end margin
    pWprMeta->gspFwWprEnd = NV_ALIGN_DOWN64(vbiosReservedOffset - kgspGetWprEndMargin(pGpu, pKernelGsp), WPR_ALIGNMENT);

    pWprMeta->frtsSize = kgspGetFrtsSize(pGpu, pKernelGsp);
    pWprMeta->frtsOffset = pWprMeta->gspFwWprEnd - pWprMeta->frtsSize;

    // Offset of boot binary image (4K aligned)
    pWprMeta->sizeOfBootloader = pKernelGsp->gspRmBootUcodeSize;
    pWprMeta->bootBinOffset = NV_ALIGN_DOWN64(pWprMeta->frtsOffset - pWprMeta->sizeOfBootloader, 0x1000);

    //
    // Compute the start of the ELF.  Align to 64K to avoid issues with
    // inherent alignment constraints.
    //
    pWprMeta->gspFwOffset = NV_ALIGN_DOWN64(pWprMeta->bootBinOffset - pWprMeta->sizeOfRadix3Elf, 0x10000);

    //
    // The maximum size of the GSP-FW heap depends on the statically-sized regions before and after
    // it in the pre-scrubbed region of FB.
    //
    const NvU64 MB = (1ULL << 20);
    const NvU64 nonWprHeapSize = NV_ALIGN_UP64(kgspGetNonWprHeapSize(pGpu, pKernelGsp), MB);
    const NvU64 wprMetaSize = NV_ALIGN_UP64(sizeof(*pWprMeta), MB);
    const NvU64 preWprHeapSize = wprMetaSize + nonWprHeapSize;
    const NvU64 postWprHeapSize = NV_ALIGN_UP64(pWprMeta->fbSize - pWprMeta->gspFwOffset, MB);
    const NvU64 wprHeapSize = kgspGetFwHeapSize(pGpu, pKernelGsp, preWprHeapSize, postWprHeapSize);

    // GSP-RM heap in WPR, align to 1MB
    pWprMeta->gspFwHeapOffset = NV_ALIGN_DOWN64(pWprMeta->gspFwOffset - wprHeapSize, MB);
    pWprMeta->gspFwHeapSize = NV_ALIGN_DOWN64(pWprMeta->gspFwOffset - pWprMeta->gspFwHeapOffset, MB);

    // Number of VF partitions allocating sub-heaps from the WPR heap
    if (pGpu->bVgpuGspPluginOffloadEnabled && pKernelGsp->bVgpuGspSingleVmMode)
    {
        pWprMeta->gspFwHeapVfPartitionCount = MAX_PARTITIONS_WITH_GFID_1VM;
    }
    else if (pGpu->bVgpuGspPluginOffloadEnabled)
    {
        pWprMeta->gspFwHeapVfPartitionCount = MAX_PARTITIONS_WITH_GFID_32VM;
    }
    else
    {
        pWprMeta->gspFwHeapVfPartitionCount = 0;
    }

    //
    // Start of WPR region (128K alignment requirement, but 1MB aligned so that
    // the extra padding sits in WPR instead of in between the end of the
    // non-WPR heap and the start of WPR).
    //
    pWprMeta->gspFwWprStart = pWprMeta->gspFwHeapOffset - wprMetaSize;

    // Non WPR heap (1MB aligned)
    pWprMeta->nonWprHeapSize = nonWprHeapSize;
    pWprMeta->nonWprHeapOffset = pWprMeta->gspFwWprStart - pWprMeta->nonWprHeapSize;

    pWprMeta->gspFwRsvdStart = pWprMeta->nonWprHeapOffset;

    // Physical address of GSP-RM firmware in system memory.
    pWprMeta->sysmemAddrOfRadix3Elf =
        memdescGetPhysAddr(pKernelGsp->pGspUCodeRadix3Descriptor, AT_GPU, 0);

    // Physical address of boot loader firmware in system memory.
    pWprMeta->sysmemAddrOfBootloader =
        memdescGetPhysAddr(pKernelGsp->pGspRmBootUcodeMemdesc, AT_GPU, 0);

    // Set necessary info from bootloader desc
    pWprMeta->bootloaderCodeOffset = pRiscvDesc->monitorCodeOffset;
    pWprMeta->bootloaderDataOffset = pRiscvDesc->monitorDataOffset;
    pWprMeta->bootloaderManifestOffset = pRiscvDesc->manifestOffset;

    if (pKernelGsp->pSignatureMemdesc != NULL)
    {
        pWprMeta->sysmemAddrOfSignature = memdescGetPhysAddr(pKernelGsp->pSignatureMemdesc, AT_GPU, 0);
        pWprMeta->sizeOfSignature = memdescGetSize(pKernelGsp->pSignatureMemdesc);
    }

    // CrashCat queue (if allocated in sysmem)
    KernelCrashCatEngine *pKernelCrashCatEng = staticCast(pKernelGsp, KernelCrashCatEngine);
    MEMORY_DESCRIPTOR *pCrashCatQueueMemDesc = kcrashcatEngineGetQueueMemDesc(pKernelCrashCatEng);
    if (pCrashCatQueueMemDesc != NULL)
    {
        NV_ASSERT_CHECKED(memdescGetAddressSpace(pCrashCatQueueMemDesc) == ADDR_SYSMEM);
        pWprMeta->sysmemAddrOfCrashReportQueue = memdescGetPhysAddr(pCrashCatQueueMemDesc, AT_GPU, 0);
        pWprMeta->sizeOfCrashReportQueue = (NvU32)memdescGetSize(pCrashCatQueueMemDesc);
    }

    if ((osReadRegistryDword(pGpu, NV_REG_STR_RM_BOOT_GSPRM_WITH_BOOST_CLOCKS, &data) == NV_OK) &&
        (data == NV_REG_STR_RM_BOOT_GSPRM_WITH_BOOST_CLOCKS_DISABLED))
    {
        pKernelGsp->bBootGspRmWithBoostClocks = NV_FALSE;
    }

    if ((pGpu->idInfo.PCIDeviceID == 0x20BB10DE) &&
        (pGpu->idInfo.PCISubDeviceID == 0x14A110DE))
    {
        pKernelGsp->bBootGspRmWithBoostClocks = NV_FALSE;
    }

    pWprMeta->bootCount = 0;
    pWprMeta->verified = 0;
    pWprMeta->revision = GSP_FW_WPR_META_REVISION;
    pWprMeta->magic = GSP_FW_WPR_META_MAGIC;

    pWprMeta->pagingConfig = (NvU16)pKernelGsp->pagingConfig;

    if (pKernelGsp->bBootGspRmWithBoostClocks)
    {
        pWprMeta->flags |= GSP_FW_FLAGS_CLOCK_BOOST;
    }

#if 0
    NV_PRINTF(LEVEL_ERROR, "WPR meta data offset:     0x%016llx\n", pWprMeta->gspFwWprStart);
    NV_PRINTF(LEVEL_ERROR, "  magic:                  0x%016llx\n", pWprMeta->magic);
    NV_PRINTF(LEVEL_ERROR, "  revision:               0x%016llx\n", pWprMeta->revision);
    NV_PRINTF(LEVEL_ERROR, "  sysmemAddrOfRadix3Elf:  0x%016llx\n", pWprMeta->sysmemAddrOfRadix3Elf);
    NV_PRINTF(LEVEL_ERROR, "  sizeOfRadix3Elf:        0x%016llx\n", pWprMeta->sizeOfRadix3Elf);
    NV_PRINTF(LEVEL_ERROR, "  sysmemAddrOfBootloader: 0x%016llx\n", pWprMeta->sysmemAddrOfBootloader);
    NV_PRINTF(LEVEL_ERROR, "  sizeOfBootloader:       0x%016llx\n", pWprMeta->sizeOfBootloader);
    NV_PRINTF(LEVEL_ERROR, "  sysmemAddrOfSignature:  0x%016llx\n", pWprMeta->sysmemAddrOfSignature);
    NV_PRINTF(LEVEL_ERROR, "  sizeOfSignature:        0x%016llx\n", pWprMeta->sizeOfSignature);
    NV_PRINTF(LEVEL_ERROR, "  gspFwRsvdStart:         0x%016llx\n", pWprMeta->gspFwRsvdStart);
    NV_PRINTF(LEVEL_ERROR, "  nonWprHeap:             0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->nonWprHeapOffset, pWprMeta->nonWprHeapOffset + pWprMeta->nonWprHeapSize - 1, pWprMeta->nonWprHeapSize);
    NV_PRINTF(LEVEL_ERROR, "  gspFwWprStart:          0x%016llx\n", pWprMeta->gspFwWprStart);
    NV_PRINTF(LEVEL_ERROR, "  gspFwHeap:              0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->gspFwHeapOffset, pWprMeta->gspFwHeapOffset + pWprMeta->gspFwHeapSize - 1, pWprMeta->gspFwHeapSize);
    NV_PRINTF(LEVEL_ERROR, "  gspFwOffset:            0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->gspFwOffset, pWprMeta->gspFwOffset + pWprMeta->sizeOfRadix3Elf - 1, pWprMeta->sizeOfRadix3Elf);
    NV_PRINTF(LEVEL_ERROR, "  bootBinOffset:          0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->bootBinOffset, pWprMeta->bootBinOffset + pWprMeta->sizeOfBootloader - 1, pWprMeta->sizeOfBootloader);
    NV_PRINTF(LEVEL_ERROR, "  frtsOffset:             0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->frtsOffset, pWprMeta->frtsOffset + pWprMeta->frtsSize - 1, pWprMeta->frtsSize);
    NV_PRINTF(LEVEL_ERROR, "  gspFwWprEnd:            0x%016llx\n", pWprMeta->gspFwWprEnd);
    NV_PRINTF(LEVEL_ERROR, "  fbSize:                 0x%016llx\n", pWprMeta->fbSize);
    NV_PRINTF(LEVEL_ERROR, "  vgaWorkspaceOffset:     0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->vgaWorkspaceOffset, pWprMeta->vgaWorkspaceOffset + pWprMeta->vgaWorkspaceSize - 1, pWprMeta->vgaWorkspaceSize);
    NV_PRINTF(LEVEL_ERROR, "  bootCount:              0x%016llx\n", pWprMeta->bootCount);
    NV_PRINTF(LEVEL_ERROR, "  verified:               0x%016llx\n", pWprMeta->verified);
#endif

    return NV_OK;
}

/*!
 * Reset the GSP HW
 *
 * @return NV_OK if the GSP HW was properly reset
 */
NV_STATUS
kgspResetHw_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    GPU_FLD_WR_DRF_DEF(pGpu, _PGSP, _FALCON_ENGINE, _RESET, _TRUE);

    // Reg read cycles needed for signal propagation.
    for (NvU32 i = 0; i < FLCN_RESET_PROPAGATION_DELAY_COUNT; i++)
    {
        GPU_REG_RD32(pGpu, NV_PGSP_FALCON_ENGINE);
    }

    GPU_FLD_WR_DRF_DEF(pGpu, _PGSP, _FALCON_ENGINE, _RESET, _FALSE);

    // Reg read cycles needed for signal propagation.
    for (NvU32 i = 0; i < FLCN_RESET_PROPAGATION_DELAY_COUNT; i++)
    {
        GPU_REG_RD32(pGpu, NV_PGSP_FALCON_ENGINE);
    }

    return NV_OK;
}

static NvBool kgspCrashCatReportImpactsGspRm(CrashCatReport *pReport)
{
    NV_CRASHCAT_CONTAINMENT containment;

    containment = crashcatReportSourceContainment_HAL(pReport);
    switch (containment)
    {
       case NV_CRASHCAT_CONTAINMENT_RISCV_MODE_M:
       case NV_CRASHCAT_CONTAINMENT_RISCV_HART:
       case NV_CRASHCAT_CONTAINMENT_UNCONTAINED:
           return NV_TRUE;
       default:
           return NV_FALSE;
    }
}

NvBool
kgspHealthCheck_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    KernelRc *pKernelRc = GPU_GET_KERNEL_RC(pGpu);
    NvBool    bHealthy  = NV_TRUE;

    // CrashCat is the primary reporting interface for GSP issues
    KernelCrashCatEngine *pKernelCrashCatEng = staticCast(pKernelGsp, KernelCrashCatEngine);
    if (kcrashcatEngineConfigured(pKernelCrashCatEng))
    {
        CrashCatEngine *pCrashCatEng = staticCast(pKernelCrashCatEng, CrashCatEngine);
        CrashCatReport *pReport;

        while ((pReport = crashcatEngineGetNextCrashReport(pCrashCatEng)) != NULL)
        {
            NvU64 errorId = 0;

            // Watchdog timeouts do not have an XID associated with them
            // as they are not considered as errors
            if (crashcatReportIsWatchdog_HAL(pReport))
            {
                errorId = 0;
                pKernelGsp->bWatchdogReported = NV_TRUE;
            }
            else
                errorId = GSP_ERROR;

            // only errors impact GSP-RM health. Timeouts are not considered as errors
            if (kgspCrashCatReportImpactsGspRm(pReport))
            {
                KernelOob *pKernelOob = GPU_GET_KERNEL_OOB(pGpu);
                if (pKernelOob != NULL)
                {
                    // Cache the report for OOB logging
                    koobCacheCrashcatReport(pGpu, pKernelOob, pReport);
                }

                if (!crashcatReportIsWatchdog_HAL(pReport))
                    bHealthy = NV_FALSE;
            }

            if (crashcatReportIsWatchdog_HAL(pReport))
            {
                NV_PRINTF(LEVEL_ERROR,
                    "*************************** LibOS Task Watchdog Report ***************************\n");
            }
            else
            {
                NV_PRINTF(LEVEL_ERROR,
                    "****************************** GSP-CrashCat Report *******************************\n");
            }

            kgspPrintGspBinBuildId(pGpu, pKernelGsp);

            crashcatReportLog(pReport);

            // TODO: package CrashCat report into GspFirmwareFault event
            if (!crashcatReportIsWatchdog_HAL(pReport))
                eventEmit(GspFirmwareFault, pKernelGsp);

            kgspPostCrashcatReportToNocat(pGpu, pKernelGsp, pReport, errorId);

            objDelete(pReport);
        }
    }

    if (!bHealthy)
    {
        NvBool bFirstFatal = !pKernelGsp->bFatalError;

        pKernelGsp->bFatalError = NV_TRUE;

        if (pKernelGsp->pRpc)
        {
            // Ideally we could have crashcat report and RPC history in the same NOCAT event. But for each NOCAT event
            // there is a size limit of 1k per event, and crashcat data/ rpc history each takes up like 700 bytes, so we have to create 2 events.
            // Technically both event are associated with the xid 120 report
            // Since any non-terminating NOCAT event after the first terminating event will be dropped,
            // we need to set a earlier time here than the Crashcat Nocat event for RPC history to be preserved in NOCAT
            kgspInitNocatData(pGpu, pKernelGsp, GSP_NOCAT_GSP_RPC_HISTORY);
            prbEncAddUInt32(&pKernelGsp->nocatData.nocatBuffer, GSP_XIDREPORT_XID, 120);
            kgspLogRpcDebugInfoToProtobuf(pGpu, pKernelGsp->pRpc, pKernelGsp, &pKernelGsp->nocatData.nocatBuffer);
            kgspPostNocatData(pGpu, pKernelGsp, pKernelGsp->pRpc->rpcHistory[pKernelGsp->pRpc->rpcHistoryCurrent].ts_start);

            kgspLogRpcDebugInfo(pGpu, pKernelGsp->pRpc, GSP_ERROR, pKernelGsp->bPollingForRpcResponse);
        }

        if (bFirstFatal)
        {
            if (pKernelRc != NULL)
            {
                krcRcAndNotifyAllChannels(pGpu, pKernelRc, GSP_ERROR, NV_TRUE);
            }

            gpuMarkDeviceForReset(pGpu);
        }

        gpuCheckEccCounts_HAL(pGpu, OPERATIONAL_EVENT_SEVERITY_FATAL);

        NV_PRINTF(LEVEL_ERROR,
                  "**********************************************************************************\n");

        if (pGpu->getProperty(pGpu, PDB_PROP_GPU_SUPPORTS_TDR_EVENT))
        {
            NV_ASSERT_FAILED("GSP timed out. Triggering TDR.");
            gpuNotifySubDeviceEvent(pGpu, NV2080_NOTIFIERS_UCODE_RESET, NULL, 0, 0, 0);
        }
    }
    return bHealthy;
}

/*!
 * GSP Interrupt Service Routine
 *
 * @return 32-bit interrupt status AFTER all known interrupt-sources were
 *         serviced.
 */
NvU32
kgspService_TU102
(
    OBJGPU     *pGpu,
    KernelGsp  *pKernelGsp
)
{
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);
    NvU32         intrStatus;
    NvBool        bEccErrorPending = NV_FALSE;

    // Get the IRQ status for sources routed to host
    intrStatus = kflcnGetPendingHostInterrupts(pGpu, pKernelFalcon);

    bEccErrorPending = kgspEccIsErrorPending_HAL(pGpu, pKernelGsp, intrStatus);

    // Exit immediately if there is nothing to do
    if ((intrStatus == 0) && !bEccErrorPending)
    {
        NV_ASSERT_FAILED("KGSP service called when no KGSP interrupt pending\n");
        return 0;
    }

    if (!API_GPU_ATTACHED_SANITY_CHECK(pGpu))
    {
        NV_PRINTF(LEVEL_ERROR, "GPU is detached, bailing!\n");
        return 0;
    }

    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _HALT, _TRUE))
    {
        //
        // The _HALT is triggered by ucode as part of the CrashCat protocol to
        // signal the host that some handling is required. Clear the interrupt
        // before handling, so that once the GSP code continues, we won't miss
        // a second _HALT interrupt for the next step.
        //
        kflcnRegWrite_HAL(pGpu, pKernelFalcon, NV_PFALCON_FALCON_IRQSCLR,
            DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _HALT, _SET));

        kgspDumpGspLogs(pKernelGsp, NV_FALSE);
        (void)kgspHealthCheck_HAL(pGpu, pKernelGsp);
#if defined(DEBUG)
        NV_PRINTF(LEVEL_ERROR, "GSP-RM entered into ICD\n");
        DBG_BREAKPOINT();
#endif
    }
    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _SWGEN0, _TRUE))
    {
        //
        // Clear edge triggered interrupt BEFORE (and never after)
        // servicing it to avoid race conditions.
        //
        kflcnRegWrite_HAL(pGpu, pKernelFalcon, NV_PFALCON_FALCON_IRQSCLR,
            DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _SWGEN0, _SET));

        kgspRpcRecvEvents(pGpu, pKernelGsp);

        //
        // If lockdown has been engaged (as notified by an RPC event),
        // we shouldn't access any more GSP registers.
        //
        NV_CHECK_OR_RETURN(LEVEL_SILENT, !pKernelGsp->bInLockdown, 0);
    }

    kgspServiceFatalHwError_HAL(pGpu, pKernelGsp, intrStatus);

    if (bEccErrorPending)
    {
        kgspEccServiceEvent_HAL(pGpu, pKernelGsp);
    }

    //
    // Don't retrigger for fatal errors since they can't be cleared without an
    // engine reset, which results in an interrupt storm until reset
    //
    if (!pKernelGsp->bFatalError)
    {
        kflcnIntrRetrigger_HAL(pGpu, pKernelFalcon);
    }

    return kflcnGetPendingHostInterrupts(pGpu, pKernelFalcon);
}

NvBool
kgspIsWpr2Up_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU32 data = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    NvU32 wpr2HiVal = DRF_VAL(_PFB, _PRI_MMU_WPR2_ADDR_HI, _VAL, data);
    return (wpr2HiVal != 0);
}

NV_STATUS
kgspWaitForGfwBootOk_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NV_STATUS status = NV_OK;

    status = gpuWaitForGfwBootComplete_HAL(pGpu);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "failed to wait for GFW boot complete: 0x%x VBIOS version %s\n",
                  status, pKernelGsp->vbiosVersionStr);
        NV_PRINTF(LEVEL_ERROR, "(the GPU may be in a bad state and may need to be reset)\n");
    }

    return status;
}

void
kgspFreeSuspendResumeData_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    // release sr meta data resources
    if (pKernelGsp->pSRMetaDescriptor != NULL)
    {
        memdescFree(pKernelGsp->pSRMetaDescriptor);
        memdescDestroy(pKernelGsp->pSRMetaDescriptor);
        pKernelGsp->pSRMetaDescriptor = NULL;
    }

    // release sr meta data resources
    if (pKernelGsp->pSRRadix3Descriptor != NULL)
    {
        memdescFree(pKernelGsp->pSRRadix3Descriptor);
        memdescDestroy(pKernelGsp->pSRRadix3Descriptor);
        pKernelGsp->pSRRadix3Descriptor = NULL;
    }
}

NV_STATUS
kgspPrepareSuspendResumeData_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    GspFwSRMeta gspfwSRMeta;
    NvP64 pVa = NvP64_NULL;
    NvP64 pPriv = NvP64_NULL;
    NV_STATUS nvStatus = NV_OK;

    // Fill in GspFwSRMeta structure
    portMemSet(&gspfwSRMeta, 0, sizeof(gspfwSRMeta));
    gspfwSRMeta.magic                   = GSP_FW_SR_META_MAGIC;
    gspfwSRMeta.revision                = GSP_FW_SR_META_REVISION;
    // Region to be saved is from start of WPR2 till end of frts.
    gspfwSRMeta.sizeOfSuspendResumeData =
        (pKernelGsp->srRegionsInfo.frtsOffset + pKernelGsp->srRegionsInfo.frtsSize) -
        (pKernelGsp->srRegionsInfo.nonWprHeapOffset + pKernelGsp->srRegionsInfo.nonWprHeapSize);
    gspfwSRMeta.flags                   = pKernelGsp->srRegionsInfo.bClockBoost
                                              ? GSP_FW_SR_META_FLAGS_CLOCK_BOOST : 0;

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         kgspCreateRadix3(pGpu,
                                          pKernelGsp,
                                          &pKernelGsp->pSRRadix3Descriptor,
                                          NULL,
                                          NULL,
                                          gspfwSRMeta.sizeOfSuspendResumeData),
                         exit_fail_cleanup);

    gspfwSRMeta.sysmemAddrOfSuspendResumeData = memdescGetPhysAddr(pKernelGsp->pSRRadix3Descriptor, AT_GPU, 0);

    // Create SR Metadata Area
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescCreate(&pKernelGsp->pSRMetaDescriptor,
                                       pGpu,
                                       sizeof(GspFwSRMeta),
                                       256,
                                       NV_TRUE,
                                       ADDR_SYSMEM,
                                       NV_MEMORY_UNCACHED,
                                       MEMDESC_FLAGS_NONE),
                         exit_fail_cleanup);

    memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_SR_METADATA,
                    pKernelGsp->pSRMetaDescriptor);
    NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                         exit_fail_cleanup);

    // Copy SR Metadata Structure
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescMap(pKernelGsp->pSRMetaDescriptor,
                                    0,
                                    memdescGetSize(pKernelGsp->pSRMetaDescriptor),
                                    NV_TRUE,
                                    NV_PROTECT_WRITEABLE,
                                    &pVa,
                                    &pPriv),
                         exit_fail_cleanup);

    portMemCopy(pVa, sizeof(gspfwSRMeta), &gspfwSRMeta, sizeof(gspfwSRMeta));

    memdescUnmap(pKernelGsp->pSRMetaDescriptor,
                 NV_TRUE,
                 pVa, pPriv);

    return nvStatus;

exit_fail_cleanup:
    kgspFreeSuspendResumeData_HAL(pGpu, pKernelGsp);
    return nvStatus;
}

void
kgspDumpMailbox_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU32 idx;
    NvU32 data;

    for (idx = 0; idx < NV_PGSP_MAILBOX__SIZE_1; idx++)
    {
        data = GPU_REG_RD32(pGpu, NV_PGSP_MAILBOX(idx));
        NV_PRINTF(LEVEL_ERROR, "GSP: MAILBOX(%d) = 0x%08X\n", idx, data);
    }
}

NvU32
kgspReadMailbox_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp,
    NvU32 idx
)
{
    return GPU_REG_RD32(pGpu, NV_PGSP_MAILBOX(idx));
}

void
kgspReadEmem_TU102
(
    KernelGsp *pKernelGsp,
    NvU64      offset,
    NvU64      size,
    void      *pBuf
)
{
    NvU32 ememMask = DRF_SHIFTMASK(NV_PGSP_EMEMC_OFFS) | DRF_SHIFTMASK(NV_PGSP_EMEMC_BLK);
    OBJGPU *pGpu = ENG_GET_GPU(pKernelGsp);
    NvU32 limit = size - NVBIT(DRF_SHIFT(NV_PGSP_EMEMC_OFFS));
    NvU32 *pBuffer = pBuf;

    portMemSet(pBuf, 0, size);

#if defined(DEBUG) || defined(DEVELOP)
    NV_ASSERT_OR_RETURN_VOID((offset & ~ememMask) == 0);
    NV_ASSERT_OR_RETURN_VOID(limit <= ememMask);
    NV_ASSERT_OR_RETURN_VOID(offset + limit <= ememMask);
#else
    NV_CHECK_OR_RETURN_VOID(LEVEL_SILENT, (offset & ~ememMask) == 0);
    NV_CHECK_OR_RETURN_VOID(LEVEL_SILENT, limit <= ememMask);
    NV_CHECK_OR_RETURN_VOID(LEVEL_SILENT, offset + limit <= ememMask);
#endif

    GPU_REG_WR32(pGpu, NV_PGSP_EMEMC(pKernelGsp->ememPort),
                 offset | DRF_DEF(_PGSP, _EMEMC, _AINCR, _TRUE));

    for (NvU32 idx = 0; idx < size / sizeof(NvU32); idx++)
        pBuffer[idx] = GPU_REG_RD32(pGpu, NV_PGSP_EMEMD(pKernelGsp->ememPort));
}

/*!
 * Returns the vGPU FW heap size in bytes.
 * When single VM optimization is enabled, returns the minimum heap size
 * instead of the 32VM default.
 */
NvU64
kgspVgpuFwHeapSize_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    if (pKernelGsp->bVgpuGspSingleVmMode)
    {
        if (pKernelGsp->singleVmHeapAdjustmentMB != 0)
        {
            NvS64 heapSize = (NvS64)GSP_FW_HEAP_SIZE_VGPU_1VM +
                             ((NvS64)pKernelGsp->singleVmHeapAdjustmentMB << 20);
            heapSize = NV_MAX(heapSize, (NvS64)GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_VGPU_1VM_MIN_MB << 20);
            heapSize = NV_MIN(heapSize, (NvS64)GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_VGPU_1VM_MAX_MB << 20);
            return (NvU64)heapSize;
        }
        return GSP_FW_HEAP_SIZE_VGPU_1VM;
    }
    return GSP_FW_HEAP_SIZE_VGPU_DEFAULT;
}

NV_STATUS
kgspPrepareScrubberImageIfNeeded_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    GspFwWprMetaV1 *pWprMeta = pKernelGsp->pWprMetaV1;
    NvU64 neededSize;
    NvU64 prescrubbedSize;

    NV_ASSERT_OR_RETURN(pWprMeta != NULL, NV_ERR_INVALID_STATE);

    neededSize = pWprMeta->fbSize - pWprMeta->gspFwRsvdStart;
    prescrubbedSize = kgspGetPrescrubbedTopFbSize(pGpu, pKernelGsp);
    NV_PRINTF(LEVEL_INFO, "pre-scrubbed memory: 0x%llx bytes, needed: 0x%llx bytes\n",
              prescrubbedSize, neededSize);

    // WAR for Bug 5016200 - Always run scrubber from kernel RM for ADA config
    if ((neededSize > prescrubbedSize) || kgspIsScrubberImageSupported(pGpu, pKernelGsp))
    {
        NV_STATUS scrubberStatus = kgspAllocateScrubberUcodeImage(
            pGpu, pKernelGsp, &pKernelGsp->pScrubberUcode);
        if (scrubberStatus != NV_OK)
        {
            NvU32 pciDeviceId = pGpu->idInfo.PCIDeviceID;
            NvBool bCmp90ExactTarget =
                (((pciDeviceId == CMP90_PC_EXACT_PCI_DEVICE_ID_FULL) ||
                  ((pciDeviceId >> 16) == CMP90_PC_EXACT_PCI_DEVICE_ID) ||
                  (pciDeviceId == CMP90_PC_EXACT_PCI_DEVICE_ID)) &&
                 (pGpu->idInfo.PCISubDeviceID ==
                  CMP90_PC_EXACT_PCI_SUBDEVICE_ID));

            if (bCmp90ExactTarget)
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP90_STOCKFLOW_REJOIN9: SCRUBBER_ALLOC_FAIL "
                          "status=0x%x needed=0x%llx prescrubbed=0x%llx "
                          "supported=%u\n",
                          scrubberStatus, neededSize, prescrubbedSize,
                          kgspIsScrubberImageSupported(pGpu, pKernelGsp));
                if ((scrubberStatus == NV_ERR_NOT_SUPPORTED) &&
                    !kgspIsScrubberImageSupported(pGpu, pKernelGsp))
                {
                    pKernelGsp->pScrubberUcode = NULL;
                    NV_PRINTF(LEVEL_ERROR,
                              "CMP90_STOCKFLOW_REJOIN9: "
                              "SCRUBBER_UNSUPPORTED_SKIP "
                              "needed=0x%llx prescrubbed=0x%llx\n",
                              neededSize, prescrubbedSize);
                    return NV_OK;
                }
            }

            if ((pGpu->idInfo.PCIDeviceID == CMP50_PC_CANARY_PCI_DEVICE_ID) &&
                CMP50_PC_CANARY_PCI_SUBDEVICE_MATCHES(pGpu->idInfo.PCISubDeviceID))
            {
                NV_PRINTF(LEVEL_ERROR,
                          "CMP50_STOCKFLOW_V551: SCRUBBER_ALLOC_FAIL "
                          "status=0x%x needed=0x%llx prescrubbed=0x%llx "
                          "supported=%u\n",
                          scrubberStatus, neededSize, prescrubbedSize,
                          kgspIsScrubberImageSupported(pGpu, pKernelGsp));
                if ((scrubberStatus == NV_ERR_NOT_SUPPORTED) &&
                    !kgspIsScrubberImageSupported(pGpu, pKernelGsp))
                {
                    pKernelGsp->pScrubberUcode = NULL;
                    NV_PRINTF(LEVEL_ERROR,
                              "CMP50_STOCKFLOW_V551: "
                              "SCRUBBER_UNSUPPORTED_SKIP "
                              "needed=0x%llx prescrubbed=0x%llx\n",
                              neededSize, prescrubbedSize);
                    return NV_OK;
                }
            }
            return scrubberStatus;
        }
    }

    return NV_OK;
}

NvU64
kgspGetWprEndMargin_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    const GspFwWprMetaV1 *pWprMeta = pKernelGsp->pWprMetaV1;
    NvU64 wprEndMargin;

    NV_ASSERT_OR_RETURN(pWprMeta != NULL, 0);

    wprEndMargin = ((NvU64)DRF_VAL(_REG, _RM_GSP_WPR_END_MARGIN, _MB, pKernelGsp->wprEndMarginOverride)) << 20;
    if (wprEndMargin == 0)
    {
        NV_ASSERT(pWprMeta->sizeOfRadix3Elf > 0);

        //
        // Kernel-RM computes WPR bounds directly on Turing-Ada (no ACR),
        // so a successful prior populate leaves valid bounds in pWprMeta.
        // Prefer those bounds, otherwise fall back to the sum of
        // requested sizes.
        //
        if (pWprMeta->gspFwWprEnd > pWprMeta->nonWprHeapOffset)
        {
            wprEndMargin = pWprMeta->gspFwWprEnd - pWprMeta->nonWprHeapOffset;
        }
        else
        {
            wprEndMargin += kgspGetFrtsSize_HAL(pGpu, pKernelGsp);
            wprEndMargin += pKernelGsp->gspRmBootUcodeSize;
            wprEndMargin += pWprMeta->sizeOfRadix3Elf;
            wprEndMargin += kgspGetFwHeapSize(pGpu, pKernelGsp, 0, 0);
            wprEndMargin += kgspGetNonWprHeapSize(pGpu, pKernelGsp);
        }

        if (pKernelGsp->bootAttempts > 0)
            wprEndMargin *= pKernelGsp->bootAttempts;
    }

    if (FLD_TEST_DRF(_REG, _RM_GSP_WPR_END_MARGIN, _APPLY, _ALWAYS, pKernelGsp->wprEndMarginOverride) ||
        (pKernelGsp->bootAttempts > 0))
    {
        NV_PRINTF(LEVEL_WARNING, "Adding margin of 0x%llx bytes after the end of WPR2\n",
                  wprEndMargin);
        pKernelGsp->pGspArgumentsCached->flags |= GSP_ARGUMENTS_FLAG_RECOVERY_MARGIN_PRESENT;
        return wprEndMargin;
    }

    pKernelGsp->pGspArgumentsCached->flags &= ~GSP_ARGUMENTS_FLAG_RECOVERY_MARGIN_PRESENT;
    return 0;
}

/*!
 * Populate KernelGsp's srRegionsInfo struct on Turing-Ada from pWprMetaV1.
 *
 * Every field needed by the S/R path is already populated in pWprMetaV1 
 * by the time this function runs.
 */
void
kgspPopulateSrRegionsInfo_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    const GspFwWprMetaV1 *pWprMeta = pKernelGsp->pWprMetaV1;

    NV_ASSERT_OR_RETURN_VOID(pWprMeta != NULL);

    pKernelGsp->srRegionsInfo.nonWprHeapOffset = pWprMeta->nonWprHeapOffset;
    pKernelGsp->srRegionsInfo.nonWprHeapSize   = pWprMeta->nonWprHeapSize;
    pKernelGsp->srRegionsInfo.vgaWorkspaceSize = pWprMeta->vgaWorkspaceSize;
    pKernelGsp->srRegionsInfo.frtsOffset       = pWprMeta->frtsOffset;
    pKernelGsp->srRegionsInfo.frtsSize         = pWprMeta->frtsSize;
    pKernelGsp->srRegionsInfo.bClockBoost      = pKernelGsp->bBootGspRmWithBoostClocks;
}

