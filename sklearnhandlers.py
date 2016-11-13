#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import pickle
from bson.binary import Binary
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class GetClasses(BaseHandler):
    def get(self):
        l=[];
        if self.db.labeledinstances.count() == 0:
            self.write_json({"error": "you have no classes bro"})
        else:
            for a in self.db.labeledinstances:
                l.append(a['label'])

            self.write_json({"classes":l})


class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        print(data)

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        # sess  = data['dsid']

        dbid = self.db.labeledinstances.insert(
            # {"feature":fvals,"label":label,"dsid":sess}
            {"feature":fvals,"label":label}
            );
        self.write_json({"id":str(dbid),"feature":fvals,"label":label})

class SetParameters(BaseHandler):
    def post(self):
        data = json.loads(self.request.body.decode("utf-8"))
        self.KNeighborsParamN = data['KNeighborsParam']
        self.RandomForestParamN = data['RandomForestParam']

class ClearDataset(BaseHandler):
    def get(self):
        self.db.labeledinstances.remove()
        self.db.models.remove()
        self.clf = {}


class UpdateModel(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        # dsid = self.get_int_arg("dsid",default=0)     ///remoivng dsid stuff

        # create feature vectors from database
        f=[];
        for a in self.db.labeledinstances.find({}):
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l=[];
        for a in self.db.labeledinstances.find({}):
            l.append(a['label'])

        # fit the model to the data
        c1 = KNeighborsClassifier(n_neighbors=self.KNeighborsParamN);
        acc1 = -1;
        c2 = RandomForestClassifier(n_estimators=self.RandomForestParamN);
        acc2 = -1;
        if l:
            c1.fit(f,l) # training
            c2.fit(f,l)
            lstar1 = c1.predict(f)
            lstar2 = c2.predict(f)
            self.clf["KNeighbors"] = c1
            self.clf["RandomForest"] = c2
            acc1 = sum(lstar1==l)/float(len(l))
            acc2 = sum(lstar2==l)/float(len(l))
            bytes1 = pickle.dumps(c1)
            bytes2 = pickle.dumps(c2)
            self.db.models.update({"classifier":"KNeighbors"}, #change to classifier
                {  "$set": {"model":Binary(bytes1)}  },
                upsert=True)
            self.db.models.update({"classifier":"RandomForest"},
                {  "$set": {"model":Binary(bytes2)}  },
                upsert=True)
            # send back the resubstitution accuracy
            # if training takes a while, we are blocking tornado!! No!!
            self.write_json({"resubAccuracyKN":acc1, "resubAccuracyRF":acc2})
        else:
            self.write_json({"error": "You have no data to train on"})
            raise HTTPError(404)

class PredictOne(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        print(data)

        vals = data['feature']
        fvals = [float(val) for val in vals]
        fvals = np.array(fvals).reshape(1, -1)
        # dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!

        # if(dsid not in self.clf):
        #     print('Loading Model From DB')
        #     tmp = self.db.models.find_one({"dsid":dsid})
        #     self.clf[dsid] = pickle.loads(tmp['model'])

        if not self.clf:
            # self.write_json({"error":"No Models have been created"})
            raise HTTPError(404)

        else:
            predLabel1 = self.clf["KNeighbors"].predict(fvals)
            predLabel2 = self.clf["RandomForest"].predict(fvals)
            self.write_json({"predictionKN":str(predLabel1), "predictionRF":str(predLabel2)})
