/*
 * livepatch_nvkm_mc_reset.c — targeted patches for K80 warm handoff.
 *
 * Patches THREE functions in nouveau.ko:
 *
 *   1. gf100_gr_fini()       — NOP: prevents PGRAPH reset on unbind that
 *                               gates GPCs and destroys warm state needed
 *                               for VFIO handoff.
 *
 *   2. nvkm_pmu_fini()       — NOP: keeps PMU falcon running across unbind.
 *                               FECS firmware depends on PMU for power
 *                               management; halting PMU causes FECS to
 *                               stall during warm boot.
 *
 *   3. nvkm_mc_disable()     — NOP: prevents ALL PMC_ENABLE bit clears
 *                               during nouveau teardown. Without this,
 *                               each subdev fini individually disables
 *                               its engine domain, collapsing PMC to
 *                               ~0xc0002020 and killing the PLL clock
 *                               tree that GPCs depend on.
 *
 * This livepatch targets .name="nouveau" so it can be loaded BEFORE nouveau.
 * When nouveau loads, these patches apply immediately.
 *
 * Load:  insmod livepatch_nvkm_mc_reset.ko    (before or after nouveau)
 * Undo:  echo 0 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled
 *        then rmmod livepatch_nvkm_mc_reset
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/livepatch.h>

/*
 * int gf100_gr_fini(struct nvkm_gr *, bool suspend)
 * Called during nouveau unbind/suspend. Resets PGRAPH via PMC, which
 * gates GPCs and halts FECS — destroying the warm state we need for
 * VFIO handoff. Return 0 (success) without touching hardware.
 */
static int livepatch_gf100_gr_fini(void *gr, bool suspend)
{
	pr_info_once("gf100_gr_fini BLOCKED (suspend=%d) — preserving GPC state for VFIO\n",
		     suspend);
	return 0;
}

/*
 * int nvkm_pmu_fini(struct nvkm_subdev *, bool suspend)
 * Called during nouveau unbind. Halts the PMU falcon, which FECS depends on
 * for GPC power management. Return 0 (success) without touching hardware.
 */
static int livepatch_nvkm_pmu_fini(void *subdev, bool suspend)
{
	pr_info_once("nvkm_pmu_fini BLOCKED (suspend=%d) — preserving PMU for VFIO\n",
		     suspend);
	return 0;
}

/*
 * void nvkm_mc_disable(struct nvkm_device *, enum nvkm_subdev_type, int)
 * Called by each subdev's fini to clear its PMC_ENABLE bit. During unbind,
 * this cascades through ~15 subdevs, collapsing PMC from 0xe011312c to
 * 0xc0002020. The PCLOCK domain (PLLs) gets disabled, killing GPC clocks
 * and making PGRAPH unable to route to GPCs after VFIO bind.
 */
static void livepatch_nvkm_mc_disable(void *device, int type, int inst)
{
	pr_info_once("nvkm_mc_disable BLOCKED (type=%d inst=%d) — preserving PMC_ENABLE for VFIO\n",
		     type, inst);
}

static struct klp_func funcs[] = {
	{
		.old_name = "gf100_gr_fini",
		.new_func = livepatch_gf100_gr_fini,
	},
	{
		.old_name = "nvkm_pmu_fini",
		.new_func = livepatch_nvkm_pmu_fini,
	},
	{
		.old_name = "nvkm_mc_disable",
		.new_func = livepatch_nvkm_mc_disable,
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
MODULE_DESCRIPTION("K80 warm handoff: block PPWR corruption, PRI IRQ storms, and GR fini reset");
