#boteic

Attempts to implement a "Back of the envelope" IceCube using public IceCube data releases

Requires nuflux and nuSQuIDS for the flux model and the propagation through the earth respectively

Rate at which particles are detected should be approximate (Estimate O(10% error)), while the energy respose is still being tested

Example extraction script corresponding to the IC86 sterile release is included as bin_mc_data.py, and intended to be run in the example/Python directory there.  It is a modified version of the script from the same directory 

deepcore_weights are based on sample_a of the 3 year study
https://icecube.wisc.edu/data-releases/2019/05/three-year-high-statistics-neutrino-oscillation-samples/

upgrade_weights from the upgrade MC
https://icecube.wisc.edu/data-releases/2020/04/icecube-upgrade-neutrino-monte-carlo-simulation/

IC86 weights from the IC86 MC for steriles
https://icecube.wisc.edu/data-releases/2016/06/search-for-sterile-neutrinos-with-one-year-of-icecube-data/
