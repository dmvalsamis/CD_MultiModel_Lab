#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 07:34:39 2022

@author: aleoikon
"""
import sys
sys.path.append(r'C:\Users\dvalsamis\change\Change_detection_SSL_Siamese')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from architectures.branch import branches_triplet
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import metrics



#distance layer according to paper
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        l1_distance = tf.reduce_sum(tf.abs(anchor - positive))
        return (ap_distance, an_distance, l1_distance)
    
#paper model
class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0) + gamma*|f(A) - f(P)|
    """

    def __init__(self, siamese_network, margin=0.2, gamma=1):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.gamma = gamma
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance, l1_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        l1_loss = self.gamma*l1_distance
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

def pretext_task_2_model(dropout, decay, IMG_HEIGHT, IMG_WIDTH , n_ch):
    base_cnn = branches_triplet(dropout, decay, IMG_HEIGHT, IMG_WIDTH , n_ch)
    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)
    embedding = Model(base_cnn.input, output, name="Embedding")
    
    anchor_input = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(n_ch)))
    positive_input = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(n_ch)))
    negative_input = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(n_ch)))

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    siamese_model = SiameseModel(siamese_network)
    
    return siamese_model, embedding
    
    