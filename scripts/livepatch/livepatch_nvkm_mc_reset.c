/*
 * livepatch_nvkm_mc_reset.c — targeted patches for warm handoff.
 *
 * STATUS: SUPERSEDED by binary-patch approach (patch_nouveau_teardown.py).
 * Kernel 6.17+ rejects R_X86_64_64 relocations with non-zero addends in
 * all out-of-tree modules, making livepatch/kprobe .ko files unloadable.
 * Retained as reference for the function signatures and rationale.
 *
 * Patches FOUR functions in nouveau.ko:
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
 *   4. nvkm_fifo_fini()      — NOP: prevents FIFO teardown which preempts
 *                               all runlists and disables PFIFO. Without
 *                               this, FECS sees "no channels" on the GR
 *                               runlist, disables GR, and halts. PFIFO
 *                               then enters an unrecoverable state.
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
	return 0;
}

static int livepatch_nvkm_pmu_fini(void *subdev, bool suspend)
{
	return 0;
}

static void livepatch_nvkm_mc_disable(void *device, int type, int inst)
{
}

static int livepatch_nvkm_fifo_fini(void *subdev, bool suspend)
{
	return 0;
}

/*
 * Kernel 6.17+ rejects R_X86_64_64 relocations with non-zero addends in
 * external modules (arch/x86/kernel/module.c: "existing value is nonzero").
 * Compile-time struct initializers for function pointers and string pointers
 * produce exactly this pattern.  Fix: declare all arrays uninitialized (BSS,
 * zero-filled) and wire them up at __init time using PC-relative code refs.
 */
static struct klp_func funcs[5];    /* [0..3] = patches, [4] = terminator */
static struct klp_object objs[2];   /* [0] = nouveau, [1] = terminator    */

static struct klp_patch patch; /* BSS — wired at __init */

static int __init livepatch_init(void)
{
	/* Wire everything at runtime — kernel 6.17 rejects R_X86_64_64
	 * relocations whose target has a non-zero existing value. */
	funcs[0].old_name = "gf100_gr_fini";
	funcs[0].new_func = livepatch_gf100_gr_fini;
	funcs[1].old_name = "nvkm_pmu_fini";
	funcs[1].new_func = livepatch_nvkm_pmu_fini;
	funcs[2].old_name = "nvkm_mc_disable";
	funcs[2].new_func = livepatch_nvkm_mc_disable;
	funcs[3].old_name = "nvkm_fifo_fini";
	funcs[3].new_func = livepatch_nvkm_fifo_fini;

	objs[0].name  = "nouveau";
	objs[0].funcs = funcs;

	patch.mod  = THIS_MODULE;
	patch.objs = objs;

	return klp_enable_patch(&patch);
}

static void __exit livepatch_exit(void)
{
}

module_init(livepatch_init);
module_exit(livepatch_exit);
MODULE_LICENSE("GPL");
MODULE_INFO(livepatch, "Y");
MODULE_DESCRIPTION("Warm handoff: NOP nouveau GR/PMU/MC/FIFO teardown to preserve GPU state for VFIO takeover");
