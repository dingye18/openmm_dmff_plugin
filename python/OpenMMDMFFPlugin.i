%module OpenMMDMFFPlugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>

%inline %{
using namespace std;
%}

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
}

%{
#include "DMFFForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <vector>
%}


/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace DMFFPlugin {

class DMFFForce : public OpenMM::Force {
public:
    DMFFForce(const string& GraphFile);
    DMFFForce(const string& GraphFile, const string& GraphFile_1, const string& GraphFile_2);

    void addParticle(const int particleIndex, const string particleType);
    void addType(const int typeIndex, const string Type);
    void addBond(const int particle1, const int particle2);
    void setPBC(const bool use_pbc);
    void setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient);

    // Extract the model info from dp model.    
    double getCutoff() const;
    int getNumberTypes() const;
    string getTypesMap() const;

    /*
    * Used for alchemical simulation. Not supported yet.
    */
    void setAlchemical(const bool used4Alchemical);
    void setAtomsIndex4Graph1(const vector<int> atomsIndex);
    void setAtomsIndex4Graph2(const vector<int> atomsIndex);
    void setLambda(const double lambda);

    /*
     * Add methods for casting a Force to a DMFFForce.
    */
    %extend {
        static DMFFPlugin::DMFFForce& cast(OpenMM::Force& force) {
            return dynamic_cast<DMFFPlugin::DMFFForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<DMFFPlugin::DMFFForce*>(&force) != NULL);
        }
    }
};

}
