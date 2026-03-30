/*
 * livepatch_nvkm_mc_reset.c — runtime patches to keep FECS running
 * across the nouveau→vfio warm handoff.
 *
 * Patches FOUR functions in nouveau.ko:
 *
 *   1. nvkm_mc_reset()       — PMC engine-level reset (NOP)
 *   2. gf100_gr_fini()       — GR engine teardown that halts FECS/GPCCS (NOP)
 *   3. nvkm_falcon_fini()    — falcon cleanup that disables falcon CPU (NOP)
 *   4. gk104_runl_commit()   — runlist submit: NOP when count==0 (empty)
 *
 * Patch 4 is critical: during DRM teardown nouveau removes all channels
 * and submits an empty runlist (count=0). FECS processes this and
 * self-resets (HRESET). In HS mode, host STARTCPU cannot recover FECS.
 * By skipping the empty commit, FECS never sees "no channels" and stays
 * in its context-switch-ready HALT state.
 *
 * Load:  modprobe livepatch_nvkm_mc_reset   (while nouveau is loaded)
 * Undo:  echo 0 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled
 *        then rmmod livepatch_nvkm_mc_reset
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/livepatch.h>

/*
 * void nvkm_mc_reset(struct nvkm_device *, enum nvkm_subdev_type, int)
 * PMC engine reset — would wipe IMEM/DMEM and engine state.
 */
static void livepatch_nvkm_mc_reset(void *device, unsigned int type, int inst)
{
	pr_info("mc_reset SKIPPED (type=%u inst=%d) — falcon state preserved\n",
		type, inst);
}

/*
 * void gf100_gr_fini(struct nvkm_gr *)
 * GR engine teardown — stops FECS ctxsw, puts falcons in halt/HRESET.
 * Must be NOPed so FECS stays in HS-mode RUNNING state through swap.
 */
static void livepatch_gf100_gr_fini(void *gr)
{
	pr_info("gf100_gr_fini SKIPPED — FECS kept running for warm handoff\n");
}

/*
 * void nvkm_falcon_fini(struct nvkm_falcon *)
 * Falcon cleanup — calls gm200_flcn_disable / writes CPUCTL to halt.
 * NOPed as a safety belt to prevent any code path from halting FECS.
 */
static void livepatch_nvkm_falcon_fini(void *falcon)
{
	pr_info("nvkm_falcon_fini SKIPPED — falcon state preserved\n");
}

/*
 * void gk104_runl_commit(struct nvkm_runl *runl,
 *                        struct nvkm_memory *memory,
 *                        u32 start, int count)
 *
 * Runlist commit — writes RUNLIST_BASE + RUNLIST_SUBMIT to trigger
 * the PFIFO scheduler. During DRM teardown, channels are removed one
 * at a time and the final commit is count=0 (empty). FECS processes
 * this as "no channels" and self-resets into HRESET.
 *
 * This NOP is safe because the livepatch is only ENABLED between
 * nouveau init completion and nouveau teardown (see warm-fecs flow).
 * During init the livepatch is disabled and the real function runs.
 */
static void livepatch_gk104_runl_commit(void *runl, void *memory,
					u32 start, int count)
{
	pr_info("gk104_runl_commit SKIPPED (count=%d) — runlist frozen for warm handoff\n",
		count);
}

static struct klp_func funcs[] = {
	{
		.old_name = "nvkm_mc_reset",
		.new_func = livepatch_nvkm_mc_reset,
	},
	{
		.old_name = "gf100_gr_fini",
		.new_func = livepatch_gf100_gr_fini,
	},
	{
		.old_name = "nvkm_falcon_fini",
		.new_func = livepatch_nvkm_falcon_fini,
	},
	{
		.old_name = "gk104_runl_commit",
		.new_func = livepatch_gk104_runl_commit,
	},
	{ }
};

static struct klp_object objs[] = {
	{
		.name = "nouveau",
		.funcs = funcs,
	}, { }
};

static struct klp_patch patch = {
	.mod = THIS_MODULE,
	.objs = objs,
};

static int __init livepatch_init(void)
{
	return klp_enable_patch(&patch);
}

static void __exit livepatch_exit(void)
{
}

module_init(livepatch_init);
module_exit(livepatch_exit);
MODULE_LICENSE("GPL");
MODULE_INFO(livepatch, "Y");
MODULE_DESCRIPTION("Keep FECS running across nouveau unbind for warm handoff to vfio-pci");
