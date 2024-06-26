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


cp -r ${HOMEDIR}/gun/gun.cpp .
cp -r ${HOMEDIR}/gun/CMakeLists.txt . 
cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/cld_steer.py .
cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/CLDReconstruction1.py .
mkdir ${DIR}/${SEED}/PandoraSettingsCLD
cp /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/PandoraSettingsCLD/* ./PandoraSettingsCLD/
cp -r ${HOMEDIR}/condor/make_pftree_clic_bindings.py .
cp -r ${HOMEDIR}/condor/tree_tools.py .
cp -r ${HOMEDIR}/gun/${GUNCARD} .

# Load key4hep 
wrapperfunction() {
    # source /cvmfs/sw.hsf.org/key4hep/setup.sh
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2024-03-07
}
wrapperfunction

# Build gun 
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make install -j 8
cd ..
./build/gun ${GUNCARD} 

# Run simulation 
ddsim --compactFile $K4GEO/FCCee/CLD/compact/CLD_o2_v05/CLD_o2_v05.xml --outputFile out_sim_edm4hep.root --steeringFile cld_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

# Run reco 
k4run CLDReconstruction1.py -n ${NEV}  --inputFile out_sim_edm4hep.root --outputBasename out_reco_edm4hep

# Create Tree 
python make_pftree_clic_bindings.py out_reco_edm4hep_edm4hep.root tree.root False False ${IDX}

mkdir -p ${OUTPUTDIR}
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py tree.root ${OUTPUTDIR}/pf_tree_${SEED}.root

