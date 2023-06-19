.. _usage:

Using Carculator Bus
====================

.. note::

   Many examples are given in this :download:`examples.zip file <_static/resources/examples.zip>` which contains a Jupyter notebook
    you can run directly on your computer.

Static vs. Stochastic mode
--------------------------

The inventories can be calculated using the most likely value of the given input parameters ("static" mode), but also using
randomly-generated values based on a probability distribution for those ("stochastic" mode). Additionally, the tool can run
one-at-a-time sensitivity analyses by quantifying the effect of incrementing each input parameter value by 10% on the end-results.

Retrospective and Prospective analyses
--------------------------------------

By default, the tool produces results across thea year 2000, 2010, 2020, 2030, 2040 and 2050.
It does so by adjusting efficiencies at the vehicle level, but also by adjusting certain aspects of the background inventories.
The latter is done by linking the vehicles' inventories to energy scenario-specific ecoinvent databases produced by ``premise``.

Export of inventories
---------------------

The library allows to export inventories in different formats, to be consumed by different tools and link to various databases.
Among the formats available, ``carculator_bus`` can export inventories as:

* Brightway2-compatible Excel file
* Simapro-compatible CSV file
* Brightway2 LCIImporter object
* Python dictionary

The inventories cna be made compatible for:

* ecoinvent 3.5 and 3.6, cut-off
* REMIND-ecoinvent produced with ``premise``
* UVEK-ecoinvent 2.2 database