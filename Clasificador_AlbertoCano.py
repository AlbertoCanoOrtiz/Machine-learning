#!/usr/bin/python
# -*- coding: utf-8 -*-


from abc import ABCMeta,abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six

class Basefes(six.with_metaclass(ABCMeta,BaseEstimator,ClassifierMixin)):
    @abstractmethod
    def predict(self,x):
        return 0
    def likelihood(self,x):
        return 0

class tfes(Basefes):

    @classmethod
    def fit(self,x,y):
        return 0

