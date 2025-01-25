Work in Progress

## Modeling tools - Project PHELMA 2024-2025

In this project we will study a process of material elaboration. The material is heated by induction. In the project,
only heat transfer will be studied. The objective is to model the heat transfer occurring in this process by using
the finite element method.

Parts 1 to 6 are here to help you write the equations, the boundary conditions, the discretized system, the mesh of
the domain, the algorithm of the resolution – for those parts you will work on paper. In part 7, you will write your
code and in part 8 you will generate the results.

Follow the method developed in the course document in order to answer to those questions.
This process of elaboration includes the following steps:

- an helical inductor. This inductor is supplied by a power generator. A medium frequency alternating current
flows through it.
- a cylindrical crucible. The material making up the crucible is not electrically conductive.
- the material to be processed, which is an electrical conductor. The geometry of the material is cylindrical.

A sinusoidal, alternative current flows through the inductor, so an alternative current is directly induced in the
material. The material is heated by Joule power density. The crucible is constituted by an insulating material so

there is no induced current in the crucible and no Joule power density in the crucible.

The dimensions of the elements of the device are given in the table 1. The physical properties of the material and
Table 1: Dimensions of the elements of the installation
|              | **Internal radius m** | **External radius m** | **Height m** |   
|:------------:|:---------------------:|:---------------------:|:------------:|
| **Crucible** |          0.1          |          0.12         |      0.4     |   
| **Material** |          0.1          |                       |      0.3     |   
| **Inductor** |          0.15         |                       |      0.2     |   

the crucible are given in the table 2. Three different materials are considered : iron, Titanium and Aluminium.
Table 2: Physical properties of the material and of the crucible
|                        | **Thermal conductivity W m−1 K−1** | **Emissivity** | **Melting temperature TfK** |
|:----------------------:|:----------------------------------:|:--------------:|:---------------------------:|
|        **Iron**        |                80.2                |      0.35      |             1811            |
|      **Titanium**      |                 15                 |      0.47      |             0.3             |
|      **Aluminium**     |                 238                |       0.1      |            933.3            |
| **Crucible (alumina)** |                 40                 |                |             2323            |

In this project only the static heat transfer has to be modeled in the material and in the crucible. The electromagnetic
phenomenon is not modeled. In the material an uniform Joule power density is imposed. The geometry
of the system is considered cylindrical. The contact between the material and the crucible is supposed perfect.

On the top surface of the material convective exchange and radiative exchange have to be modeled. Convective
exchange takes place on the external boundaries of the crucible.

The values of room temperature Tr and convective exchange coefficient H are :
- Tr = 300K 
- Hcv = 20W/m2/K
