# Quantifying Cooperation and Allostery in Protein-DNA Complexes

We present a method for quantifying internal allosteric networks within proteins using transfer entropy, a statistical measure derived from Schreiber (2000) used for measuring the statistical coherence between time-series. Using the trajectory of alpha carbons in molecules as time-series data, we are able to deduce aspects of the fluctuational dynamics within proteins.

## Model Systems

A set of 2 protein-DNA complexes were utilized in this study: the gal and lac operons bound to their respective repressors. Supposedly, fluctuations originating from the hinge-helix region cause a distinct "bend" in the DNA. Molecular dynamics (MD) simulations were performed and Gaussian Network Models (GNM) were calculated for these complexes and their mis-matches (gal repressor bound to lac operon and vice versa). The hope is that transfer entropy will reveal influential residues elucidating the origin of DNA bending.

## Transfer Entropy

Transfer entropy, a information-theoretic measure, is an alternative to Granger causality, to measure how much uncertainty the past of one time-series (the source) reducing about the past of another time series (the destination). This is done by conditioning on histories of the past and destination time-series. Essentially, if the past source and destination time series reduces more information than the past destination time series, then the source is said the "influence" or "granger-cause" the destination time series. As an alternative Granger causality, transfer entropy is viable for application to non-linear systems. Originally for discrete data, the transfer entropy has been adapted to estimate influences present in continuous data using various estimation methods such as histogram binning, kernel destination, and k nearest neighbors (knn) estimation. Here, we focus on the knn approach first proposed by Kraskov, Stogbauer, and Grassberger (KSG), as it is naturally bias-canceling and has been shown to reduce resulting variance.

## Normalized Transfer Entropy

Classically, transfer entropy estimates dictate how many bits or nats (as dictated by the base of the logarithm) of information the source variable reduces about the destination variable. Since different systems are subject to different dynamics, they may differ in the amount of entropy which can be reduced. Consequently, if one wants to quantitatively compare estimates between systems, it is necessary to normalize transfer entropy estimates. In the discrete sense this can be done by either dividng by the entropy of the destination variable or by dividing the conditional entropy of the destination given its past. Since KSG transfer entropy takes a nearest neighbors approach, normalization is much more difficult. We propose to normalize the transfer entropy by feature-scaling it between a maximum and minimum theoretical transfer entropy. These extremes are obtained using a cardinality approach that optimizes the number of points in the respective subspaces in the full joint space containing the future and past destination variables and the past source variable (For more information, [click here!](https://github.com/benjaminAhlbrecht/ProteinDNAResearch/files/6300235/NormalizedTransferEntropy.pdf)).

## Constructing Information Flow Networks

Since transfer entropy quantifies how much uncertainty (entropy) one variable reduces in another variable, it is often referred to a measure of "information flow" between systems. When calculating information flow in large networks (such as proteins), one can simply calculate the transfer entropy between each pair of nodes in the network, creating a pairwise transfer entropy matrix. Additionally, to determine the direction of information flow between two nodes, it is natural to simply subtract the information flow between the two opposing directions. This is equivalent to subtracting the transposed pairwise matrix from its original. Last, it is often useful 

## Future Work

Transfer entropy appears promising in the analysis of the fluctuation dynamics present in proteins. While it is insufficient for determining the significance of specific residues, it is able to show influential regions within protein-DNA complexes such as the hinge-helices. Regardless, much more work must be done to effectively evaluate an the information-theoretic basis of transfer entropy. Some ideas are discussed below:

1. It is possible to condition on other possible sources to remove redundant information or add synergistic information. While it is computationally unrealistic to test all sources within a large network, it may be possible to test other potential sources of a given destination.
2. Fluctuational dynamics cannot be explained using the trajectory of atoms alone. It may be helpful to conditional on other variables such as local potential energies within the residues in the protein.
3. An enhanced graph-theory analysis may effectively cluster influential residues together. Possible options may be a minimum spanning tree (MST) or planar maximally filtered graph (PMFG), which have been shown to work effectively with partial correlations. Since transfer entropy can be easily expressed as a difference between mutual informations, it is entirely possible to create a distance metric to effectively represent transfer entropy networks visually.
