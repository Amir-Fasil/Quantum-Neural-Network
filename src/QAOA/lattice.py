import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

class ConceptLattice():

    def __init__(self, concepts: list, context):

        self.context = context
        self.conceptLattice = concepts
        self.length = len(concepts)  # I can use this to quantify the number of node I need.

    def get_number_of_concepts(self):
        return self.length

    def get_concept_lattice(self):

        """This just returns a list of the concepts."""
        return self.conceptLattice

    def get_proper_concept(self):

        """This function retruns a list of concepts that are 
        tuples instead of Concept objects
        """
        proper_concepts = []
        for concept in self.conceptLattice:
            proper_concepts.append(concept.get_Concept())

        return proper_concepts
    

    def basis_attribute(self, attributes):
        
        intent_power_set = self.context.get_powerSet(attributes)
        bassis_attributes = set()
        for elements in intent_power_set:
            #print("Elements:", elements)
            if len(elements) == 0 and len(elements) == len(attributes):
                continue
            first_derv = self.context.Differentiate(elements)
            second_derv = self.context.Differentiate(first_derv)
            #print("Second Derivative:", second_derv)
            if elements != second_derv:   # This checks if A = A'' and for an attribute to be a basis attribute it must not be equal to its second derivative.
                 # They need not to be a single attribute they can be a subset of the total attributes.
                sub_power_set = self.context.get_powerSet(elements)
                psdeo_intent_flag = True
                for sub_elements in sub_power_set:
                    if len(sub_elements) == 0 and len(sub_elements) == len(elements):
                        continue
                    sub_first_derv = self.context.Differentiate(sub_elements)
                    sub_second_derv = self.context.Differentiate(sub_first_derv)
                    #print(f"Sub Element: {sub_elements}, Sub Second Derivative: {sub_second_derv}, and Elements: {elements}")
                    if sub_second_derv.issubset(elements):
                        psdeo_intent_flag = False
                        break

                if psdeo_intent_flag:
                    bassis_attributes = bassis_attributes.union(elements)
                    #print("TRUE")

        return bassis_attributes

    def set_cover(self):

        """This function is ment to give necessary varibales for building 
        QUBO model to solve minimum set cover for our concept lattice """
        
        ############ Cost Matrix ############
        intents = self.context.get_intents()

        basis_attributes = self.basis_attribute(intents)  # Custom defined cost
        extent_sets = [concept.get_extent() for concept in self.conceptLattice]
        corresponding_intents = [concept.get_intent() for concept in self.conceptLattice]
        set_cost = [(len(basis_attributes) - len(basis_attributes.intersection(S))) for S in corresponding_intents]
        A = max(set_cost) + 1
        corresponding_concepts = self.conceptLattice  # A list of concept objects
        universal_extent = set.union(*extent_sets) if extent_sets else set()
        element_contained = [len(universal_extent.intersection(extent_sets[i])) for i in range(len(extent_sets))]
        Q = np.zeros((len(corresponding_concepts), len(corresponding_concepts)))
        for i in range(len(corresponding_concepts)):
            for j in range(len(corresponding_concepts)):
                if i == j:
                    Q[i][j] = set_cost[i] - A * element_contained[i]
                else:
                    element_shared = len(extent_sets[i].intersection(extent_sets[j]))
                    Q[i][j] = A * element_shared

        return Q
    
    def get_lattice(self):

        """This function returns the concept lattice(Graphs)."""

        concepts = self.get_proper_concept()
        concept_nodes = [(frozenset(extent), frozenset(intent)) for (extent, intent) in concepts]

        G = nx.Graph()
        G.add_nodes_from(concept_nodes)

        for i, (e1, i1) in enumerate(concept_nodes):
            for j, (e2, i2) in enumerate(concept_nodes):
                if i != j and i1 > i2 and len(i1) == len(i2) + 1:
                    G.add_edge((e1, i1), (e2, i2))

        levels = defaultdict(list)
        for concept in concept_nodes:
            level = len(concept[1]) 
            levels[level].append(concept)
 
        pos = {}
        for level, nodes_at_level in sorted(levels.items(), reverse=True):  
            spacing = 1.0
            width = (len(nodes_at_level) - 1) * spacing
            for i, node in enumerate(nodes_at_level):
                x = i * spacing - width / 2
                y = level
                pos[node] = (x, y)


        labels = {n: f"E:{set(n[0])}\nI:{set(n[1])}" for n in G.nodes()}
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightcoral', node_size=1800, font_size=9)
        plt.title("Concept Lattice (Hasse Diagram, Ordered by Intent)")
        plt.gca().invert_yaxis()  
        plt.show()
        
        


