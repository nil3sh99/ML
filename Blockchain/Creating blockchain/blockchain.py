#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 02:21:03 2018

@author: nilesh
"""

#importing libraries
import datetime
# because each block will have its own time
import hashlib
# because we have to hash
import json
# because we have to use dump function
from flask import Flask, jsonify
# we will import flask class, and to return messagees with
# postman app... to display results


# Building a blockchain
            hash_operation =  hashlib.sha256((new_proof**2 - previous_proof**2).encode()).hexdigest() 

class blockchain:    
    def __init__(self):
        self.chain = [] # chain containing the chains --> list
        self.create_block(proof = 1, previous_hash= '0')
                                    #SHA256 can only use encoded STRINGS and hence we took it as a string
    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) +1,
                 'timeStamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash':previous_hash
                 # we can add any data to the blockchain by adding any key here
                 }
        self.chain.append(block) # chain is a list and hence we used its append function
        return block
    def get_previous_block(self):
        return self.chain[-1]
    
    def proof_of_work(self, previous_proof):
        new_proof = 1 # to solve the prob we have to increment this as a base
        check_proof = False
        while check_proof is False:
            hash_operation =  hashlib.sha256((new_proof**2 - previous_proof**2).encode()).hexdigest() 
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                check_proof = False
                new_proof += 1
        return new_proof
    
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys = True).encode()
        return haslib.sha256(encoded_block).hexdigest()
    
    def is_chain_valid(self, chain):
        prev_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(prev_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation =  hashlib.sha256((proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                return False
            previous_block = block
            block_index += 1
            
        return True
    
# Part2 - MIning the blockchain

app = Flask(__name__)

# crating our blockchain
Blockchain = blockchain()
    
    
    
    

            
            
            