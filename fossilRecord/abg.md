Generating conformational energy landscapes to validate molecular docking approach

To this point
I've used AutoDock Vina for molecular docking of a random selection of monosaccharide ring conformations into the active site, and see that conformations associated with the catalytic itinerary are better binders. Conformational energy landscapes are far more resource intensive than molecular docking, so I want to prove that the docking approach resembles the result obtained from more intense calculation methods.

Work
Generate conformational energy landscapes of the monosaccharide ring conformation inside of the active site of a selection of carbohydrate-active enzymes (CAZymes), for example by selection of representative members of the Glycosyl Hydrolase families
Perform the simpler molecular docking approach and compare

Possible tangents
Validate the ring conformation generator

Expected outcome
Docking works well as replacement for conformational energy landscapes in some subset of CAZymes

Reference
https://pubs.acs.org/doi/10.1021/jacs.5b01156 (esp. fig.10)
#project-proposals  •  5/10/2026
rosīe
 pinned a message to this channel. See all pinned messages. — 5/20/2026 10:49 PM
Tamison — 5/21/2026 12:01 PM
It seems like this tech will need/use GROMACS?    Thats new to me, but I've worked with KOKKOS in teh past.   I plan to have standard modern nvidia gpu, but if we come across any genreational diffrences I have some HMB2 cards on hand I can use
Alistaire — 5/21/2026 1:30 PM
I do think Gromacs would be fine, but I'm not familiar enough with it to know if its enhanced sampling methods are good for carbohydrates.
Wei-Tse Hsu
Hands-on tutorials: Enhanced sampling methods using GROMACS | Wei-T...
For those of you who just dabbled in the field of enhanced sampling, I hope this mini-course can save you several days (or more) of time getting onboard.
Hands-on tutorials: Enhanced sampling methods using GROMACS | Wei-T...
Mark — 5/21/2026 5:12 PM
I'm interested in helping out on this. I have access to GROMACS and some GPU's through NSF (I use Texas A&M's ACES for MD sims). I can include you in my access, as long as US-based and has a .edu if additional compute is needed.
Tamison — 5/21/2026 5:17 PM
Lovely.  So we will be able to tier the work, and I can run a across a couple gpu,  adn then for later stages escalated to funded grade HPC.   Those A100 type cards are absolute beasts, and I'd love to see the scale increase on wn actual workload
Alistaire — 5/21/2026 6:13 PM
Currently at the 2009-2011 era of this kind of work, seems they use metadynamics packages to sample the cremer-pople θ and φ as CVs using a carbohydrate forcefield like GROMOS 45A4. I think more recently you'd use Gromacs06.
Alistaire — 5/23/2026 3:41 PM
So I notice almost every protein-carbohydrate Free Energy Landscape (FEL) is being sampled by some kind of metadynamics on, most often, both QM/MM and MD. You often see Amber used for MD, which is openly available -- I'm not sure whether that allows metadynamics. The QM/MM part either uses CPMD, HYPERCHEM or GROMETA (a GROMACS package for metadynamics), with most commonly the PBE DFT for QM and forcefields GLYCAM06, FF99SB for MM with TIP3P water model.
Tamison — 5/23/2026 10:29 PM
This domain is new to me, but the shape of the compute/validation work is familiar. I’ve been working in hotSpring/barraCuda on reference → Rust → GPU validation layers, including MD/QCD-level primitives, precision work, and cross-vendor shader kernels.

I already have some of the lower-level primitives that seem relevant here. If you can point at the right reference tools/papers/workflows, i can start ingesting them, mapping what overlaps, and evolving the missing primitives for this next layer/domain.
Alistaire — 5/24/2026 12:22 AM
So just to see I understand; barraCUDA is a non-vendor package for compiling GPU software to CUDA, then MD/QCD I presume quantum chromodynamics seems to encode maybe some quantum physics equations into that GPU architecture. Presumably a primitive is an equation of state or an interaction energy analogue for physics particles.

So you're asking for the system of equations of the quantum mechanics part like PBE DFT? At the computational chemistry level I haven't quite found some publication that discusses this in great detail; but there's the Alonso-Gil 2019 thesis (download) I think is closest to discussing the equations (Chapter 2.2-2.4). The best is probably to look at those of the original PBE, GLYCAM, Amber forcefield publications if you're planning to reimplements them? 
My interpretation of the Free Energy Landscape stuff is you need to either do QM/MM to get a transition state (worth a publication) or you do metadynamics to nudge the system of equations into different low, mid and high energy states by adding 'fake energy terms' to keep certain dihedrals a certain way. QM/MM is you running QM and a forcefield at the same time, then reconciling energy differences between the two. This takes some extra fuckery to do. Metadynamics is nudging the energies a way that you want, which is also kinda nonstandard if you're dealing with primitives/equations.

Or maybe I misinterpret what GPU compilation you're planning on doing for QM/MM?
Alistaire — 5/24/2026 12:31 AM
Interesting, KOKKOS is apparently how LAMMPS HPC is compiled (arxiv), which does molecular dynamics. LAMMPS also has a metadynamics code for collective variables (online manual) that you'd need for sampling the dihedrals.

GROMACS is compiled C++17 (per Google Search) and there's some forum posts saying they use it in HPCs (like this) so perhaps that's also reasonable? 
Alistaire — 5/24/2026 12:42 AM
Section 5.4 of the colvars LAMMPS manual "When Colvars is enabled, atomic coordinates are collected on a single CPU core, where collective variables and their biases are computed. This means that in the case of simulations that are already being run over large numbers of nodes, or inside a GPU, a Colvars calculation may produce a significant overhead." 
Tamison — 5/24/2026 6:43 AM
i can start working with that.   a lot of the issue is that the language, the compile and the framework arent united.


but for the barraCuda, yes ive been rewriting all the CUDA shaders, cutting the legacy, and then extending and validating on physics to fp64,  including layering fp32 cores as DF64.

df64 is 10^-12 i believe to fp64 full precision (10^-14?),  and every consumer card is 90% fp32
but i got hotQCD running cuda free.  32^3 x 48 with flavors on a 3090
Tamison — 5/24/2026 6:54 AM
but the process starts by using industry standard tools,  so this week is getting gromacs (modern branch first) stable and running local
Tamison — 5/24/2026 7:23 AM
So what we are doing here,  findign industry tools,  and running them on consumer grade is teh first (ans often harder solve),  and acts as a control and valiadtion target for teh ai to abstract patterns from.
barraCuda is univeral math (true precision),  coralReef is teh compiler (im working on an IR idempotency loop for passing any compiled drivers from vendor to agnostic),  and tehn toadStool is teh hardware disptach, allowing me to passthoru the gpu as if it was vfio, while having tehh kernel like access to teh chip.

so any and all math is wsgl shader,  with precison raising and lowering based on stable math algos.   then via toadStool tells coralReef what hardware is availble and dispathc accorgnly.    so gpu vs cpu is 100% dispatch issue,  math is portable across hardware 
Tamison — 5/24/2026 7:51 AM
So im not sure whhat colvars is,  but I know hardware dispatch.    Generally systems want to split work across gpu adn cpu.    so for example when solving theh long running qcd issues,  you can stream work to teh gpu, and hhave it only stream resulst back to teh cpu,  so the gpu is never idle,  adn teh cpu side (where we collect teh data) can continue work async.

A lot of code makes bad dispatchh assumptions as they hahve been tuned adn evovled in hpc grade environments.  so its about scaling it down, and tehn abstracting.

modern consumer mobos also have other fun ssytems,  like you can pass work between gpu if you ahve multiple in a ssytemsusing jsut tehh pcie bandwidht,    and a slice of ram can be used to shortcut around teh cpu to stop teh bottle necks of transfer speeds
Tamison — 5/24/2026 8:04 AM
What we already have (production-validated, f64 precision, consumer GPU):

Lennard-Jones 6-12 + cell list neighbor finding
Coulomb direct + PPPM (Ewald) long-range
Velocity Verlet integrator
Nose-Hoover, Langevin, Berendsen thermostats
Verlet + cell list neighbor lists (runtime-adaptive)
PBC (cubic, orthorhombic)
Full observable suite (RDF, VACF, MSD, SSF, stress tensor, heat current)
f64 + DF64 (emulated double on fp32 cores, ~10⁻¹² precision)
GPU shader compilation to all NVIDIA (SM35–SM120) and AMD (GCN5/RDNA) targets
Sovereign GPU dispatch via VFIO (no CUDA dependency)
219 experiments, 500+ quantitative checks, proven on RTX 3090 / Titan V / RTX 5060


What's missing (the chemistry layer on top of the physics):

Harmonic bond — V(r) = ½k(r - r₀)²
Harmonic angle — V(θ) = ½k(θ - θ₀)²
Dihedral torsion — V(φ) = Σ kₙ(1 + cos(nφ - δₙ))
Improper dihedral — V(ψ) = ½k(ψ - ψ₀)²
Force field parameter/topology reader (GROMOS 45a4 / GLYCAM06)
Metadynamics bias (Cremer-Pople CV + Gaussian hill deposition)
So the full nonbonded MD engine is done — we're adding 4 bonded force field shaders, a topology reader for carbohydrate force fields, and the metadynamics layer. GROMACS 2026.0 is installed locally as the industry control so I can parity-check against it as I go.
What we already have (production-
6 Messages ›
Tamison
2d ago
Jeremy — 5/24/2026 9:38 AM
Harmonic bond — V(r) = ½k(r - r₀)²
Harmonic angle — V(θ) = ½k(θ - θ₀)²
Dihedral torsion — V(φ) = Σ kₙ(1 + cos(nφ - δₙ))
Improper dihedral — V(ψ) = ½k(ψ - ψ₀)²
how do u get superscripts and subscripts on discord?? 🙏 
Alistaire — 5/24/2026 9:50 AM
It seems they are the unicode super/subscript characters
Tamison — 5/24/2026 11:46 AM
Lol, yeah thats all ai.   I realized with LaTeX it was faster for me to descirbe a word problem,    formatiign is trivial to teh ai.   they are unicode,  but it can do markdown easily as well
Alistaire — 5/24/2026 12:10 PM
Hm is the whole thing running in an agent environment so the "5/5 checks passed" for example could mean nothing?
Tamison — 5/24/2026 12:13 PM
compeltely reasonable for you to ask.     and Ill begin to find out towards the end on my side as I eovel to mroe agnostic code.   but teh 5/5 checks if from teh ai reviewing the wokr,  not teh ai work itself.    i tend to run 3-4 ide per computer (1 per sub project,  so coralReef, toadStool, and barraCuda are primal teams)   and teh hotspring is running math from valditon agaisnt estblished
but im working on getting teh full data to you, and that will be a far superioir valdaitoin
math being correct is trivial,   interpration beign correct is not
Alistaire — 5/24/2026 12:14 PM
Right fair enough, I guess we can see what's cooked up at the end
Tamison — 5/24/2026 1:38 PM
Attachment file type: archive
pseudoSpore_cazyme_fel_v0.6.0.tar.gz
107.30 KB
Tamison — 5/24/2026 1:42 PM
Work is still in flight, and Im still working on some of teh packaging piplines,  adn have a ring ordering question as well  (blockers 1 and 2 are local links and packaging issues im solving):
"""
The critical item for Alistaire:

Finding 3 is the blocker. The free xylose FEL has its global minimum at θ=172° (labeled 1C4), but 4C1 is the expected ground state for β-D-xylopyranose. Until Alistaire confirms the Cremer-Pople atom ordering convention (C1-C2-C3-C4-C5-O5), the Module 2 landscape can't be interpreted against Iglesias-Fernández 2015, and Module 3 results won't be meaningful either. The artifact correctly flags this as the action item — that's good.
"""
Tamison — 5/24/2026 2:27 PM
It took me some time to udnerstand what you meant here as well.    I still get misdaignoses and erros form time to time as misgernations occur,   but I built teh hardware to have tehh ai local via IDE like cursor or whatever youd like and compile on my infra.   so its more like pair programming
Alistaire — 5/24/2026 4:19 PM
Yea I'm going through the files:
enzyme-bound-puckering
 
The puckering indices in plumed.dat are completely wrong; the actual atom indices are 6599, 6600, 6601, 6602, 6603, 6607 for C1, C2, C3, C4, C5, O5
I'm not sure what's done but xylose_m1.pdb is a direct copy of the input structure; it has φ 203.029 θ 86.550 and Q 0.817 which are different from θ 172
I don't think it made a water box or anything around the PDB file; actually the PDB file is nowhere

xylose-puckering-fel
 
The file xylose_charmm.pdb isn't xylose, it's actually β-D-Lyxose in a 1C4 pucker with φ 107.096 θ 9.919 and Q 0.529
It says it's a FEL in water but from what I can tell it's done in vacuum?

I guess the xylose puckering (using lyxose) get reported as the enzyme bound being 1C4? 
I don't really know what run, but maybe it's good to see something returned a bunch of angles and energies?
Tamison — 5/24/2026 4:52 PM
Im on the road so ill go through and refine when im home.

so this has been a rapid sprint, AND its mostly in python for the gromacs,   which is the least reliable for AI.

specific things that are missing are likely the pipline automation work still evolving.

however water box vs vacuum and others, these are things fir me to reeximne and work through.


really did through this together this morning, I usually focus on specific sub tasks and usulky have a lineage of papers i can validate the methods against.
Tamison — 5/24/2026 5:03 PM
i like to treat it as a knowledgeable but somewhat unfocused student.

thanks for taking a look at the data,  your input will allow me to dive into the domain as well (water box vs vacuum, and the struggles of MD and QM/MM).
so that I can be much more effective with my evolution.

this really was all new to me before the last few days, and this definitely helps me navigate.

That was a round 1 prototype, and should definitely be the least refined going forward
Alistaire — 5/24/2026 5:06 PM
Sure, I know of some papers that describe stuff a bit better and have steps in-between simulations that maybe are useful if you want to have as simple of a test of replicating a result. Though it's mostly something you get in QM/MM
Tamison — 5/25/2026 8:14 AM
please link them and ill get to work.    reprodcuing papers is helpful as tehy ten to descibe and connect ssytems that tutorials leave out.
and for teh last prototype you were looking at.   Most of it was my error, not teh ai.    it did run in water,  I do ahve the pdb,   i pulled teh trigger and sent before all data was in it,  adn also didnt share some data as i thought it to be regenrable.

teh xylose vs lyxose was a legitimate ai error,  but I had never heard of lyxose so I didnt catch it.  and I belive one of teh downloads when we valdiated was differtn conformation tahn teh metadata suggested,  so upstream issues as well.


The next one will have a lot more of the data patterns now that I udnerstand teh data and problem a little better.   was the error of teh crafstmen (me) , not teh tools lol.


if there is already an existing standard taht woul help too.   like if you ahd a collegue on teh otherside of teh world with teh compute,  and you were corresponding;  what would teh data records passing back and forth look like?
Alistaire — 5/25/2026 10:57 AM
I think the most thorough descriptions are generally in the theses from the Rovira group, one being Nin-Hill, 2020 (ub.edu download from Scholar) like Chapter 3. I don't know enough about its methods, but it seems to be using full DFT even for the vacuum FEL of galactose (section 3.4.2.1 and tables 3.3, 3.4). It uses CPMD however, but perhaps the information from those is enough to replicate in GROMACS
CPMD is apparently freely available on Github since 2022 - https://github.com/CPMD-code/CPMD/tree/main
Alistaire — 5/25/2026 2:50 PM
Hm interesting, there's PLUMED-NEST which contains files to replicate published outputs(?) that I search for carbohydrate and there's 4 hits. They don't seem to be FELs but it's nice to know it should be replicable 
Alistaire — 5/25/2026 5:32 PM
And the google group might be an amazing source for GROMACS-PLUMED like some questions about a pyranose sugar FEL
Tamison — 5/25/2026 6:23 PM
alright,  this should be muchh more substantial.   It inlcueds tehh data, pdb, and figures that should help us get the data transmittable.    Ill start looking at tehh groups and links you dropped as I build out the valditon systems,   adn tehn move to modeling things for teh porject
Attachment file type: archive
pseudoSpore_hotSpring-CAZyme-FEL_v1.5.0.tar.gz
4.72 MB
Tamison — 5/25/2026 8:09 PM
Yep,  this is 100% what I needed to understand whhat you need from me.  I am begign to ingest a bunch to careaet more baseline valdidations and tune my data deployment chassis.      My near to mid term goal is to evolve a FEL to submit
