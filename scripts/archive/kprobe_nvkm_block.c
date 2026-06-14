/*
 * kprobe_nvkm_block.c — kprobe-based FIFO teardown blocker.
 *
 * STATUS: SUPERSEDED by binary-patch approach (coral-driver::tools::elf_patcher).
 * Kernel 6.17+ rejects out-of-tree kprobe modules. The ELF binary patcher
 * NOP's all 4 teardown functions directly in nouveau.ko at the machine-code
 * level. Retained as reference for the kprobe technique.
 *
 * Loaded AFTER nouveau init completes to block nvkm_fifo_fini during unbind.
 * The livepatch handles gf100_gr_fini, nvkm_pmu_fini, and nvkm_mc_disable
 * (safe to NOP during both init and unbind). But nvkm_fifo_fini cannot be
 * NOP'd during init — it performs essential FIFO setup. This kprobe module
 * is loaded post-init so it only intercepts the unbind-time call.
 *
 * Load:  insmod kprobe_nvkm_block.ko    (after nouveau init, before unbind)
 * Unload: rmmod kprobe_nvkm_block
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/ptrace.h>

static int blocked_fifo_fini;

static int pre_nvkm_fifo_fini(struct kprobe *p, struct pt_regs *regs)
{
	if (!blocked_fifo_fini) {
		blocked_fifo_fini = 1;
		pr_info("nvkm_fifo_fini BLOCKED — preserving PFIFO/FECS for VFIO\n");
	}
	regs->ax = 0;
	regs->ip = *(unsigned long *)regs->sp;
	regs->sp += sizeof(unsigned long);
	return 1;
}

static struct kprobe kp_fifo_fini = {
	.symbol_name = "nvkm_fifo_fini",
	.pre_handler = pre_nvkm_fifo_fini,
};

static int __init kprobe_block_init(void)
{
	int ret = register_kprobe(&kp_fifo_fini);
	if (ret < 0) {
		pr_err("register_kprobe(nvkm_fifo_fini) failed: %d\n", ret);
		return ret;
	}
	pr_info("armed: nvkm_fifo_fini blocked for warm VFIO handoff\n");
	return 0;
}

static void __exit kprobe_block_exit(void)
{
	unregister_kprobe(&kp_fifo_fini);
	pr_info("disarmed: fifo=%d calls blocked\n", blocked_fifo_fini);
}

module_init(kprobe_block_init);
module_exit(kprobe_block_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("kprobe-based nvkm_fifo_fini blocker for warm VFIO handoff");
