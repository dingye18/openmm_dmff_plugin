/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "ReferenceDMFFKernels.h"
#include "DMFFForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include <typeinfo>
#include <iostream>
#include <map>
#include <algorithm>
#include <limits>

using namespace DMFFPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcDMFFForceKernel::~ReferenceCalcDMFFForceKernel(){
    delete &jax_model;
    if(used4Alchemical){
        delete &jax_m1;
        delete &jax_m2;
    }
    return;
}

void ReferenceCalcDMFFForceKernel::initialize(const System& system, const DMFFForce& force) {
    graph_file = force.getDMFFGraphFile();
    type4EachParticle = force.getType4EachParticle();
    typesIndexMap = force.getTypesIndexMap();
    used4Alchemical = force.alchemical();
    forceUnitCoeff = force.getForceUnitCoefficient();
    energyUnitCoeff = force.getEnergyUnitCoefficient();
    coordUnitCoeff = force.getCoordUnitCoefficient();

    natoms = system.getNumParticles();
    coord_shape[0] = natoms;
    coord_shape[1] = 3;
    exclusions.resize(natoms);

    // Load the ordinary graph firstly.
    jax_model = cppflow::model(graph_file);
    if(used4Alchemical){
        cout<<"Used for alchemical simulation. Load the other two graphs here."<<endl;
        graph_file_1 = force.getGraph1_4Alchemical();
        graph_file_2 = force.getGraph2_4Alchemical();
        jax_m1 = cppflow::model(graph_file_1);
        jax_m2 = cppflow::model(graph_file_2);
        lambda = force.getLambda();
        atomsIndex4Graph1 = force.getAtomsIndex4Graph1();
        atomsIndex4Graph2 = force.getAtomsIndex4Graph2();
        natoms4alchemical[1] = atomsIndex4Graph1.size();
        natoms4alchemical[2] = atomsIndex4Graph2.size();
        
        // pair<int, int> stores the atoms index in U_B. This might be useful for force assign.
        atomsIndexMap4U_B = vector<pair<int,int>>(natoms4alchemical[1] + natoms4alchemical[2]);

        // Initialize the input and output array for alchemical simulation.
        dener4alchemical[1] = 0.0;
        dforce4alchemical[1] = vector<VALUETYPE>(natoms4alchemical[1] * 3, 0.);
        dvirial4alchemical[1] = vector<VALUETYPE>(9, 0.);
        dcoord4alchemical[1] = vector<VALUETYPE>(natoms4alchemical[1] * 3, 0.);
        dbox4alchemical[1] = vector<VALUETYPE>(9, 0.);
        dtype4alchemical[1] = vector<int>(natoms4alchemical[1], 0);
        
        for(int ii = 0; ii < natoms4alchemical[1]; ++ii){
            int index = atomsIndex4Graph1[ii];
            atomsIndexMap4U_B[index] = make_pair(1, ii);
            dtype4alchemical[1][ii] = typesIndexMap[type4EachParticle[index]];
        }
        coord_shape_1[0] = natoms4alchemical[1];
        coord_shape_1[1] = 3;
        
        dener4alchemical[2] = 0.0;
        dforce4alchemical[2] = vector<VALUETYPE>(natoms4alchemical[2] * 3, 0.);
        dvirial4alchemical[2] = vector<VALUETYPE>(9, 0.);
        dcoord4alchemical[2] = vector<VALUETYPE>(natoms4alchemical[2] * 3, 0.);
        dbox4alchemical[2] = vector<VALUETYPE>(9, 0.);
        dtype4alchemical[2] = vector<int>(natoms4alchemical[2], 0);
        
        for(int ii = 0; ii < natoms4alchemical[2]; ++ii){
            int index = atomsIndex4Graph2[ii];
            atomsIndexMap4U_B[index] = make_pair(2, ii);
            dtype4alchemical[2][ii] = typesIndexMap[type4EachParticle[index]];
        }
        coord_shape_2[0] = natoms4alchemical[2];
        coord_shape_2[1] = 3;


        if ((natoms4alchemical[1] + natoms4alchemical[2]) != natoms){
        //cout<<natoms4alchemical[1]<<" "<<natoms4alchemical[2]<<" "<<natoms<<endl;
        throw OpenMMException("Wrong atoms number for graph1 and graph2. Summation of atoms number in graph 1 and 2 is not equal to total atoms number.");
        }
    }

    // Initialize the ordinary input and output array.
    // Initialize the input tensor.
    dener = 0.;
    dforce = vector<VALUETYPE>(natoms * 3, 0.);
    dvirial = vector<VALUETYPE>(9, 0.);
    dcoord = vector<VALUETYPE>(natoms * 3, 0.);
    dbox = vector<VALUETYPE>(9, 0.);
    dtype = vector<int>(natoms, 0);    
    // Set atom type;
    for(int ii = 0; ii < natoms; ii++){
        // ii is the atom index of each particle.
        dtype[ii] = typesIndexMap[type4EachParticle[ii]];
    }

    AddedForces = vector<double>(natoms * 3, 0.0);
    
}

double ReferenceCalcDMFFForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    // Extract the box size.
    if ( ! context.getSystem().usesPeriodicBoundaryConditions()){
        dbox = {}; // No PBC.
        throw OpenMMException("No PBC is not supported yet.");
    }
    Vec3* box = extractBoxVectors(context);
    // Transform unit from nanometers to required units for DMFF model input.
    dbox[0] = box[0][0] * coordUnitCoeff;
    dbox[1] = box[0][1] * coordUnitCoeff;
    dbox[2] = box[0][2] * coordUnitCoeff;
    dbox[3] = box[1][0] * coordUnitCoeff;
    dbox[4] = box[1][1] * coordUnitCoeff;
    dbox[5] = box[1][2] * coordUnitCoeff;
    dbox[6] = box[2][0] * coordUnitCoeff;
    dbox[7] = box[2][1] * coordUnitCoeff;
    dbox[8] = box[2][2] * coordUnitCoeff;
    auto dbox_tensor = cppflow::tensor(dbox, box_shape);
    
    // Set input coord.
    for(int ii = 0; ii < natoms; ++ii){
        // Multiply by 10 means the transformation of the unit from nanometers to angstrom.
        dcoord[ii * 3 + 0] = pos[ii][0] * coordUnitCoeff;
        dcoord[ii * 3 + 1] = pos[ii][1] * coordUnitCoeff;
        dcoord[ii * 3 + 2] = pos[ii][2] * coordUnitCoeff;
    }
    auto dcoord_tensor = cppflow::tensor(dcoord, coord_shape);

    // Assign the input coord for alchemical simulation.
    if(used4Alchemical){
        // Set the input coord and box array for graph 1 first.
        for(int ii = 0; ii < natoms4alchemical[1]; ii ++){
            int index = atomsIndex4Graph1[ii];
            dcoord4alchemical[1][ii * 3 + 0] = pos[index][0] * coordUnitCoeff;
            dcoord4alchemical[1][ii * 3 + 1] = pos[index][1] * coordUnitCoeff;
            dcoord4alchemical[1][ii * 3 + 2] = pos[index][2] * coordUnitCoeff;
        }
        dbox4alchemical[1] = dbox;
        auto dcoord_tensor4alchemical1 = cppflow::tensor(dcoord4alchemical[1], coord_shape_1);

        // Set the input coord and box array for graph 2.
        for(int ii = 0; ii < natoms4alchemical[2]; ii ++){
            int index = atomsIndex4Graph2[ii];
            dcoord4alchemical[2][ii * 3 + 0] = pos[index][0] * coordUnitCoeff;
            dcoord4alchemical[2][ii * 3 + 1] = pos[index][1] * coordUnitCoeff;
            dcoord4alchemical[2][ii * 3 + 2] = pos[index][2] * coordUnitCoeff;
        }
        dbox4alchemical[2] = dbox;
        auto dcoord_tensor4alchemical2 = cppflow::tensor(dcoord4alchemical[2], coord_shape_2);
    }

    computeNeighborListVoxelHash(
        neighborList, 
        natoms,
        pos,
        exclusions,
        box, 
        true, 
        1.2,
        0.0
    );
    int totpairs = neighborList.size();
    std::vector<int32_t> pairs_v;
    for (int ii = 0; ii < totpairs; ii++)
    {
        int32_t i1 = neighborList[ii].second;
        int32_t i2 = neighborList[ii].first;
        pairs_v.push_back(i1);
        pairs_v.push_back(i2);
    }
    pair_shape[0] = totpairs;
    pair_shape[1] = 2;
    auto pair_tensor = cppflow::tensor(pairs_v, pair_shape);

    auto output = jax_model({{"serving_default_args_tf_0:0", dcoord_tensor}, {"serving_default_args_tf_1:0", dbox_tensor}, {"serving_default_args_tf_2:0", pair_tensor}}, {"PartitionedCall:0", "PartitionedCall:1"});

    dener = output[0].get_data<ENERGYTYPE>()[0];
    dforce = output[1].get_data<VALUETYPE>();

    if (used4Alchemical){
        throw OpenMMException("Alchemical is not supported yet.");
    }

    if(used4Alchemical){
        throw OpenMMException("Alchemical is not supported yet.");
    } else{
        // Transform the unit from eV/A to KJ/(mol*nm)
        for(int ii = 0; ii < natoms; ii ++){
            AddedForces[ii * 3 + 0] = dforce[ii * 3 + 0] * forceUnitCoeff;
            AddedForces[ii * 3 + 1] = dforce[ii * 3 + 1] * forceUnitCoeff;
            AddedForces[ii * 3 + 2] = dforce[ii * 3 + 2] * forceUnitCoeff;
        }
        // Transform the unit from eV to KJ/mol
        dener = dener * energyUnitCoeff;
    }

    if(includeForces){
        for(int ii = 0; ii < natoms; ii ++){
        force[ii][0] += AddedForces[ii * 3 + 0];
        force[ii][1] += AddedForces[ii * 3 + 1];
        force[ii][2] += AddedForces[ii * 3 + 2];
        }
    }
    if (!includeEnergy){
        dener = 0.0;
    }
    // Return energy.
    return dener;
}


