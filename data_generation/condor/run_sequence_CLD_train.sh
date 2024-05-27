#!/bin/bash

HOMEDIR=${1}
GUNCARD=${2}
NEV=${3}
SEED=${4}
OUTPUTDIR=${5}
DIR=${6}
IDX=${7}
mkdir ${DIR}
mkdir ${DIR}/${SEED}
cd ${DIR}/${SEED}


# cp -r ${HOMEDIR}/gun/gun.cpp .
# cp -r ${HOMEDIR}/gun/compile_gun.x .
# cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/cld_steer.py .
# cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/CLDReconstruction1.py .
# mkdir ${DIR}/${SEED}/PandoraSettingsCLD
# cp /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/PandoraSettingsCLD/* ./PandoraSettingsCLD/
cp -r ${HOMEDIR}/condor/make_pftree_clic_bindings.py .
cp -r ${HOMEDIR}/condor/tree_tools.py .
# cp -r ${HOMEDIR}/gun/${GUNCARD} .
# echo ${HOMEDIR}/gun/${GUNCARD}
# echo ${INPUTFILE}

#first recompile gun locally 
echo "  "
echo " ================================================================================ "
echo "  "
echo "compiling gun"
echo "  "
echo " ===============================================================================  "
echo "  "

wrapperfunction() {
    # source /cvmfs/sw.hsf.org/key4hep/setup.sh
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2024-03-07
}
wrapperfunction


# source compile_gun.x


# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "running gun"
# echo "  "
# echo " ===============================================================================  "
# echo "  "
# ./gun ${GUNCARD} 

# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "gun complete ..."
# echo "  "
# echo " ================================================================================ "
# echo "  "


# # # source /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/key4hep-stack/2024-03-10-gidfme/setup.sh
# # #When you want to revert the changes:


# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "run simulation ..."
# echo "  "
# echo " ================================================================================ "
# echo "  "
# ddsim --compactFile $K4GEO/FCCee/CLD/compact/CLD_o2_v05/CLD_o2_v05.xml --outputFile out_sim_edm4hep.root --steeringFile cld_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "run reconstruction ..."
# echo "  "
# echo " ================================================================================ "
# echo "  "

# k4run CLDReconstruction1.py -n ${NEV}  --inputFiles out_sim_edm4hep.root --outputBasename out_reco_edm4hep

echo "  "
echo " ================================================================================ "
echo "  "
echo "produce pf tree ..."
echo "  "
echo " ================================================================================ "
echo "  "

python make_pftree_clic_bindings.py out_reco_edm4hep_edm4hep.root tree.root False False ${IDX}

echo "  "
echo " ================================================================================ "
echo "  "
echo "copying output file  ..."
echo "  "
echo " ================================================================================ "
echo "  "

mkdir -p ${OUTPUTDIR}
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py tree.root ${OUTPUTDIR}/pf_tree_${SEED}.root

echo "  "
echo " ================================================================================ "
echo "  "
echo "file copied ... "
echo "  "
echo " ================================================================================ "
echo "  "