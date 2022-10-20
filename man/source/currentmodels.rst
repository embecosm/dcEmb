.. _currentmodels:

Current Models
==============

We currently implement two DCM models, a COVID-19 model and a 3Body model.

dcm_covid
##########

This code implements a version of the "Dynamic Causal Modelling of COVID-19"
`model <https://www.fil.ion.ucl.ac.uk/spm/covid-19/>`_. This code follows the
reference implementation in the development version of SPM12 fairly closely, and
provides a point of comparison between the dcEmb C++ implementation, and the 
version implemented in SPM12. 

dcm_3body
#########

This code implements a Dynamic Causal Model over a classical physics example,
the three body problem. This code serves both as a useful example of how a
simple DCM model can be implemented, and an important sanity check of our
software against a known result.

This model is based on the `three body problem <https://en.wikipedia.org/
wiki/Three-body_problem>`_, the problem of taking the initial position and 
velocities of three point masses and solving their subsequent motion according 
to newtons laws of motion.  The three body problem is often given as an example 
of a simple physical system that doesn't have a closed form solution to it's
equations of motion. Though there are not closed form solutions to the three 
body problem, it is known to have stable solutions, in which the thee bodies
move in a periodic orbit. It is one of these stable solutions (a figure-of-8) 
that this example is based on.

The premise of this model is that, given the position and velocities of one of
three planets from our known stable figure-of-8 orbit over a time period, and 
given purposely incorrect priors on the initial positions and velocities of the
other two planets, the DCM framework should be able to recover posteriors that
are close to the true initial conditions that lead to a stable orbit. This works 
fairly well. We find that we can set priors that themselves lead to systems that
are extremely different to our stable orbit and still recover initial condtions
that are very close to it. We do note that there is a limit to this though -
providing priors that are sufficiently far away from thetrue solution makes
convergence impossible.

Currently this models only tests the model inversion stage. Testing against
model selection is a work in progress.

.. raw:: html
    
    <video width="640" height="640" controls src="_static/3body.mp4"></video>

*Lightest colors: true stable orbits. Darkest Colors: posterior mean orbits.
Middle colors: prior mean orbits. The orbit of the red planet over the time
period was used to recover the orbits of the blue and green ones. The (darkest
color) posterior mean orbits are almost directly under the true orbits, and may
be a bit difficult to see without zooming in.*