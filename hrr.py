# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:18:53 2014

@author: bptripp
"""
import numpy as np

def bind(a, b, do_normalize=True):
    """
    Arguments: 
    a: a vector
    b: another vector of the same length
    c: vectors a and b 'bound' by circular convolution 
    """

    fa = np.fft.fft(a, axis=0)
    fb = np.fft.fft(b, axis=0)
    bound = np.real(np.fft.ifft(fa*fb, axis=0))
    if do_normalize:
        return normalize(bound)
    else:
        return bound

def unbind(c, b):
    """
    Arguments: 
    b: a vector 
    c: two vectors a and b 'bound' by circular convolution 
    Returns:
    aa: an approximation of the vector a
    """

    fc = np.fft.fft(c, axis=0)
    fb = np.conj(np.fft.fft(b, axis=0))
    unbound = np.real(np.fft.ifft(fc*fb, axis=0))
    return unbound

def normalize(a): 
    """ Normalizes a vector to length 1 (this should be done with composite HRRs.) 

    Arguments: 
    a: a vector
    Returns:
    aa: normalized to unit length
    """
    
    result = a
    n = np.linalg.norm(a)
    if n > 0:
        result = a / n
    return result
    
def random_vectors(dim, n): 
    """
    Generate n random unit vectors of length dim. 
    """
    v = np.random.randn(dim, n)
    norms = np.sqrt(np.sum(v**2,axis=0))
    return np.divide(v, np.tile(norms, (dim, 1))) #normalize to length 1
    

class CleanupMemory: 
    """
    Cleanup memory for holographic reduced representations. See Plate (2003).     
    An HRR is just a vector. HRRs can be bound and unbound using circular 
    convolution. Unbinding is the approximate inverse of binding, so bound
    concepts can approximately be retrieved from a complex composite concept. 
    However, the process is lossy. After a few such steps the results are badly
    degraded. So, results are often compared to a list of known vectors, and 
    replaced with the most similar vector in the list. The cleanup memory
    stores the list and supports various operations related to it, like 
    cleaning up, adding a new known vector to the list, etc. 
    """
    
    def __init__(self, concepts, dim):
        """
        Arguments: 
        dim - Dimension of the vectors (probably >250). 
        concepts - List of concept names. 
        """        
        self.dim = dim
        self.concepts = concepts
        self.vectors = random_vectors(dim, len(concepts))
        self.mappings = {}

    def similarity(self, vector):
        """
        Arguments: 
        similarities - Vector of inner products between current output and each concept. 
        """
        return np.dot(self.vectors.transpose(), vector)
    
    def all_similarities(self):
        """
        Arguments: 
        similarities - Vector of inner products between each pair of concepts. 
        """
        return np.dot(self.vectors.transpose(), self.vectors)
    
    def cleanup(self, vector):
        """
        Vector - A vector to clean up. 
        Index - Index of the most similar concept to given vector. 
        Clean - Cleaned up output. 
        """
        if len(self.concepts) == 0: 
            return (None, None, None)
            
        similarities = np.dot(self.vectors.transpose(), vector)
        max_similarity = np.max(similarities)
        index = list(similarities == max_similarity).index(True)
        clean = self.vectors[:,index]
        return (index, clean, max_similarity)

    #TODO: not sure this is right -- retest
    def move_toward(self, to_index, from_index, inner_product):
        """Rotates a concept vector closer to another one. 
        
        Arguments: 
        to_index - Index of vector to move closer to. 
        from_index - Index of vector to be moved.        
        """
        to = self.vectors[:,to_index]
        frm = self.vectors[:,from_index]
        orthogonal = frm - to.transpose()*np.dot(frm, to)
        moved = np.dot(inner_product, to) + np.sin(np.arccos(inner_product))*orthogonal
        self.vectors[:,from_index] = moved
    
    def get(self, concept):
        """
        Arguments:         
        concept - String label for a concept
        
        Returns: 
        vector - Vector associated with concept
        """
        index = self.get_index(concept)
        return self.vectors[:,index]
        
    def get_bound(self, concept):
        """
        Arguments: 
        concept - String label for a concept or a binding of multiple labels (e.g. 'cat*ran*fast')
        """
        result = None
        for part in concept.split('*'): 
            index = self.get_index(part)
            vector = self.vectors[:,index]
            if result is None: 
                result = vector
            else: 
                result = bind(result, vector)
                
        return result
        
    def get_mapping(self, key_vector): 
        """
        Arguments: 
        key_vector - Vector of the key of a mapping. 
        
        Returns: 
        Vector of value associated with a sufficiently similar cleaned up key 
        vector, or zero vector if there is no such mapping. 
        """
        (index, clean, max_similarity) = self.cleanup(key_vector)
        if max_similarity > .8:
            key_label = self.concepts[index]
            value_label = self.mappings.get(key_label)
            value_vector = self.get(value_label)
        else: 
            value_vector = np.zeros(self.dim)
            
        return value_vector
    
    def add_mapping(self, key, value): 
        """
        Adds a dictionary-like lookup relationship between two concepts. The 
        concepts must be part of this memory. 
        
        Arguments: 
        key - Label of key concept. 
        value - Label of value concept. 
        """
        assert key in self.concepts
        assert value in self.concepts
        self.mappings[key] = value
    
    def get_index(self, concept):
        """
        Arguments: 
        concept: string label for a concept
        index: index at which concept is stored
        """
        if not concept in self.concepts: 
            raise Exception('The concept %s is not in this memory' % concept)
            
        return self.concepts.index(concept)
    
    def replace(self, concept, vector):
        """
        Replaces an existing vector with a new one. 
        
        concept: a concept label
        vector: a new vector to be used for the concept 
        """
        index = self.get_index(concept)
        self.vectors[:,index] = vector

    def add(self, concept, vector=None, normalize_vector=True):
        """
        Add a specific vector (probably composed from others). 
        
        Arguments: 
        concept - String label. 
        vector - Associated vector. 
        """
        if vector is None:
            vector = random_vectors(self.vectors.shape[0], 1)
        
        if len(vector.shape) == 1:
            vector = np.expand_dims(vector, 1)
        
        if normalize_vector: 
            vector = normalize(vector)
        
        self.concepts.append(concept)
        self.vectors = np.append(self.vectors, vector, axis=1)


#TODO (later): consider indexing by (egocentric) time and/or space; creating 
#   more room by driving other nearby concepts apart, e.g. moveToward(-new_vector)
class ShortTermMemory(CleanupMemory): 
    """
    A cleanup memory that is meant for frequent addition of concepts which may be 
    relatively transient. New concepts overwrite existing concepts that are sufficiently
    similar. 
    """
    
    def __init__(self, dim, removal_threshold=0.75): 
        """
        Arguments: 
        dim - Dimension of the vectors. 
        removal_threshold - If an added vector has or higher inner product 
            with the closest existing vector the existing one is removed.
        """
        CleanupMemory.__init__(self, [], dim)
        self.removal_threshold = removal_threshold
        self.keys = np.zeros((dim,0)) 
        self.past = random_vectors(dim, 1)[:,0]

    def add(self, concept, vector, key=None):
        """
        Adds a specific vector, normally composed of other vectors. If a similar
        vector already exists it will be removed to make room for this one. 
        
        Arguments: 
        concept - String concept label. 
        vector - Vector encoding of concept. 
        key - Optional key vector with which to associate concept. 
        """
        vector = normalize(vector)
        
        (index, clean, max_similarity) = self.cleanup(vector)
        if max_similarity >= self.removal_threshold: 
            self._delete(index)
                
        CleanupMemory.add(self, concept, vector)
        
        if key is None: 
            key = np.zeros((self.dim,1))
        elif len(key.shape) == 1: 
            key = np.expand_dims(key, 1)

        self.keys = np.append(self.keys, key, axis=1)

        
    def _delete(self, index): 
        del self.concepts[index]
        self.vectors = np.delete(self.vectors, index, axis=1)
        self.keys = np.delete(self.keys, index, axis=1)
        
            
    def remove_by_key(self, key): 
        """
        Removes entries with sufficiently similar keys. 
        """
        similarities = np.dot(self.keys.transpose(), key)
        
        to_remove = [i for (i, val) in enumerate(similarities) if val>self.removal_threshold]
        to_remove.sort(reverse=True)
        
        for i in range(len(to_remove)): 
            self._delete(to_remove[i])
            
    
    #TODO: maybe we don't need this -- not using "past" states for anything
    def move_to_past(self, vector): 
        """
        Flags information as non-current. It is replaced with bind(self.past, vector). 
        
        Arguments: 
        vector - Vector to identify as non-current. 
        """
        (index, clean, max_similarity) = self.cleanup(vector)
        if max_similarity > .9: 
            new_concept = 'past.%s' % self.concepts[index]
            new_vector = bind(self.vectors[:,index], self.past)
            del self.concepts[index]
            self.vectors = np.delete(self.vectors, index, axis=1)
            self.add(new_concept, new_vector)
