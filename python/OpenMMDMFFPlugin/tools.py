from __future__ import absolute_import
try:
    from openmm import app, KcalPerKJ
    import openmm as mm
    from openmm import unit as u
    from openmm.app import *
    import openmm.unit as unit
except:
    from simtk import unit as u
    import simtk.openmm as mm
    from simtk.openmm.app import *
    import simtk.openmm as mm
    import simtk.unit as unit
    
import sys
from datetime import datetime, timedelta
try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib is not installed.")
import numpy as np

try:
    string_types = (unicode, str)
except NameError:
    string_types = (str,)

from .OpenMMDMFFPlugin import DMFFForce

class ForceReporter(object):
    def __init__(self, file, group_num, reportInterval):
        self.group_num = group_num
        if self.group_num is None:
            self._out = open(file, 'w')
            #self._out.write("Get the forces of all components"+"\n")
        else:
            self._out = open(file, 'w')
            #self._out.write("Get the forces of group "+str(self.group_num) + "\n") 
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        # return (steps, positions, velocities, forces, energies)
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        if self.group_num is not None:
            state = simulation.context.getState(getForces=True, groups={self.group_num})
        else:
            state = simulation.context.getState(getForces=True)
        forces = state.getForces().value_in_unit(u.kilojoules_per_mole/u.nanometers)
        self._out.write(str(forces)+"\n")



def DrawScatter(x, y, name, xlabel="Time", ylabel="Force, unit is KJ/(mol*nm)", withLine = True, fitting = False):
    plt.clf()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(name)

    color_list = ['r', 'g', 'b']

    if len(y.shape) > 1 and y.shape[1] != 0:
        for ii, y_row in enumerate(y):
            plt.scatter(x, y_row, c=color_list[ii], alpha=0.5)
            if withLine:
                plt.plot(x, y_row)    
    else:
        plt.scatter(x, y, c='b', alpha=0.5)
        if withLine:
            plt.plot(x, y)
    if fitting:
        coef, bias = np.polyfit(x, y, 1)
        min_x = min(x)
        max_x = max(x)
        fitting_x = np.linspace(min_x, max_x, 500)
        fitting_y = coef * fitting_x + bias
        plt.plot(fitting_x, fitting_y, '-r')

    plt.savefig("./output/"+name+'.png')
    return



class DMFFModel():
    def __init__(self, model_file, model_file_1 = None, model_file_2 = None) -> None:
        self.model_file = model_file
        self.dmff_force = DMFFForce(model_file)
        #self.cutoff = self.dp_force.getCutoff()
        #self.numb_types = self.dp_force.getNumberTypes()
        #self.type_map_raw = self.dp_force.getTypesMap()
        #self.type_map_dict, self.dp_model_types = self.__decode_type_map(self.type_map_raw)
        self.IsAlchemical = False
        
        if model_file is not None and model_file_1 is not None and model_file_2 is not None:
            del self.dmff_force
            self.dmff_force = DMFFForce(model_file, model_file_1, model_file_2)
            self.model_file_1 = model_file_1
            self.model_file_2 = model_file_2
            self.IsAlchemical = True

        # Set up the atom type
        #for atom_type in self.type_map_dict.keys():
        #    self.dp_force.addType(self.type_map_dict[atom_type], atom_type)
        
        return
    
    def __decode_type_map(self, type_map_string):
        type_map_dict = dict()
        type_list = type_map_string.split()
        for ii, atom_type in enumerate(type_list):
            type_map_dict[atom_type] = ii
        dp_model_types = list(type_map_dict.keys())
        
        assert len(dp_model_types) == self.numb_types, "Number of types is not consistent with numb_types from dp model"
        
        return type_map_dict, dp_model_types
    
    def setUnitTransformCoefficients(self, coordinatesCoefficient, forceCoefficient, energyCoefficient):
        """_summary_

        Args:
            coordinatesCoefficient (_type_): Coefficient for input coordinates that transforms the units of the coordinates from nanometers to the units used by the DP model.
            forceCoefficient (_type_): Coefficient for forces that transforms the units of the DP-predicted forces from the units used by the DP model to kJ/(mol * nm).
            energyCoefficient (_type_): Coefficient for energies that transforms the units of the DP-predicted energy from the units used by the DP model to kJ/mol.
        """
        self.dmff_force.setUnitTransformCoefficients(coordinatesCoefficient, forceCoefficient, energyCoefficient)
        return
    
    def createSystem(self, topology, particleNameLabeler = "element", particles_group_1 = None, particles_group_2 = None, Lambda = None):
        """_summary_

        Args:
            topology (_type_): OpenMM Topology object
            particleNameLabeler (str, optional): labeler of atom type in topology, element or atom_name. Defaults to "element".
            particles_group_1 (list, optional): list of particle index in group 1. Defaults to None. Used for alchemical free energy calculation.
            particles_group_2 (list, optional): list of particle index in group 2. Defaults to None. Used for alchemical free energy calculation.
            Lambda (float, optional): lambda value for alchemical free energy calculation. Defaults to None.
        
        """
        dmff_system = mm.System()
        
        # Add particles into force.
        for atom in topology.atoms():
            if particleNameLabeler == "element":
                atom_type = atom.element.symbol
            elif particleNameLabeler == "atom_name":
                atom_type = atom.name
            #if atom_type not in self.dp_model_types:
            #    raise Exception(f"Atom type {atom_type} is not found in {self.dp_model_types}.")
            
            dmff_system.addParticle(atom.element.mass)
            self.dmff_force.addParticle(atom.index, atom_type)
            
        
        # Add bond information into DMFFForce for the PBC issue 
        # during the trajectory saving. 
        for bond in topology.bonds():
            self.dmff_force.addBond(bond[0].index, bond[1].index)
        
        if self.IsAlchemical:
            raise Exception("Alchemical free energy calculation is not supported for DMFF model.")
            if particles_group_1 is None:
                raise Exception("particles_group_1 is required for alchemical DP")
            if particles_group_2 is None:
                raise Exception("particles_group_2 is required for alchemical DP")
            if Lambda is None:
                raise Exception("Lambda is required for alchemical DP")
            
            #print(f"{len(particles_group_1)} particles are selected for group 1 in alchemical simulation.")
            #print(f"{len(particles_group_2)} particles are selected for group 2 in alchemical simulation")
            
            self.dp_force.setAtomsIndex4Graph1(particles_group_1)
            self.dp_force.setAtomsIndex4Graph2(particles_group_2)
            self.dp_force.setLambda(Lambda)
        
        dmff_system.addForce(self.dmff_force)
        
        return dmff_system