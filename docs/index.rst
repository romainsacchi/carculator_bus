.. image:: /_static/img/mediumsmall_2.png
   :align: center

.. raw:: html

    <p align="center">
      <a href="https://badge.fury.io/py/carculator-truck" target="_blank"><img src="https://badge.fury.io/py/carculator-truck.svg"></a>
      <a href="https://github.com/romainsacchi/carculator_bus" target="_blank"><img src="https://github.com/romainsacchi/carculator_bus/actions/workflows/main.yml/badge.svg?branch=master"></a>
      <a href="https://ci.appveyor.com/project/romainsacchi/carculator_bus" target="_blank"><img src="https://ci.appveyor.com/api/projects/status/github/romainsacchi/carculator_bus?svg=true"></a>
      <a href="https://coveralls.io/github/romainsacchi/carculator_bus" target="_blank"><img src="https://coveralls.io/repos/github/romainsacchi/carculator_bus/badge.svg"></a>
      <a href="https://carculator_bus.readthedocs.io/en/latest/" target="_blank"><img src="https://readthedocs.org/projects/carculator_bus/badge/?version=latest"></a>
     </p>


.. _intro:

Carculator Bus
==============

``carculator_bus`` is a parameterized model that allows to generate and characterize life cycle inventories for different bus configurations, according to selected:

* powertrain technologies (9): diesel engine, electric motor, hybrid, plugin-hybrid, etc.,
* year of operation (2): 2000, 2010, 2020, 2030, 2040 and 2050 (with the possibility to interpolate in between)
* and sizes: 3.5t, 7.5t, 18t, 26t, 40t and 60t

The methodology used to develop ``carculator_bus`` is explained in an article :cite:`ct-1074`.
The tool has a focus on buses.

At the moment, the tool has a focus on the transport of dry goods.

More specifically, ``carculator_bus`` generates `Brightway2 <https://brightway.dev/>`_ and
`SimaPro <https://www.simapro.com/>`_ compatible inventories, but also directly provides characterized results against several midpoint and endpoint indicators from the impact assessment method *ReCiPe 2008 (mid- and endpoint)* and *ILCD 2.0 2018 (only midpoint)* as well as life cycle cost indicators.

``carculator_bus`` differentiates itself from other bus LCA models as it uses time- and energy-scenario-differentiated background inventories for the future, resulting from the coupling between the `ecoinvent database <https://ecoinvent.org>`_ and the scenario outputs of PIK's integrated assessment model `REMIND <https://www.pik-potsdam.de/research/transformation-pathways/models/remind/remind>`_, using the `premise <https://github.com/romainsacchi/premise>`_ library.
This allows to perform prospective study while consider future expected changes in regard to the production of electricity, cement, steel, heat, etc.

Objective
---------

The objective is to produce life cycle inventories for vehicles in a transparent, comprehensive and quick manner,
to be further used in prospective LCA of transportation technologies.

Why?
----

Many life cycle assessment (LCA) models of transport vehicles exist. Yet, because LCA of vehicles, particularly
for electric battery vehicles, are sensitive to assumptions made in regards to electricity mix used for charging,
lifetime of the battery, load factor, trip length, etc., it has led to mixed conclusions being published in the
scientific literature. Because the underlying calculations are kept undocumented, it is not always possible to
explain the disparity in the results given by these models, which can contribute to adding confusion among the public.

Because ``carculator_bus`` is kept **as open as possible**, the methods and assumptions behind the generation of
results are easily identifiable and adjustable. Also, there is an effort to keep the different modules (classes)
separated, so that improving certain areas of the model is relatively easy and does not require changing extensive
parts of the code. In that regard, contributions are welcome.

Finally, beside being more flexible and transparent, ``carculator_bus`` provides interesting features, such as:

* a stochastic mode, that allows fast Monte Carlo analyses, to include uncertainty at the vehicle level
* possibility to override any or all of the 200+ default input vehicle parameters (e.g., load factor, drag coefficient) but also calculated parameters (e.g., driving mass).
* hot pollutants emissions as a function of the driving cycle, using `HBEFA <https://www.hbefa.net/e/index.html>`_ 4.1 data, further divided between rural, suburban and urban areas
* noise emissions, based on `CNOSSOS-EU <https://ec.europa.eu/jrc/en/publication/reference-reports/common-noise-assessment-methods-europe-cnossos-eu>`_ models for noise emissions and an article by :cite:`ct-1015` for inventory modelling and mid- and endpoint characterization of noise emissions, function of driving cycle and further divided between rural, suburban and urban areas
* export of inventories as an Excel/CSV file, to be used with Brightway2 or Simapro, including uncertainty information. This requires the user to have `ecoinvent` installed on the LCA software the bus inventories are exported to.
* export inventories directly into Brightway2, as a LCIImporter object to be registered. Additionally, when run in stochastic mode, it is possible to export arrays of pre-sampled values using the `presamples <https://pypi.org/project/presamples/>`_ library to be used together with the Monte Carlo function of Brightway2.
* development of an online graphical user interface (in progress): `carculator online <https://carculator.psi.ch>`_

Get started with :ref:`Installation <install>` and continue with an overview about :ref:`how to use the library <usage>`.

User's Guide
------------

.. toctree::
   :maxdepth: 2

   installation
   usage
   modeling
   structure
   validity

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

.. toctree::
   :maxdepth: 2
   :hidden:

   references/references
   annexes
