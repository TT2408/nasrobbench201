# -*- coding: utf-8 -*-

import json, os

class NasRobBench201:
    ############################################################################
    def __init__(self, metapath,datasetpath):
        
        with open(metapath) as f:
            self.meta = json.load(f)
            self.map_str_to_id = {m["nb201-string"]:k for k,m in self.meta["ids"].items()}
            self.non_isomorph_ids = [i for i, d in self.meta["ids"].items() if d["isomorph"]==i]

        with open(datasetpath) as f:
            self.res = json.load(f)

    def get_uid(self, i):
        """
        Returns the evaluated architecture id (if given id is isomorph to another network)
        
        Parameters
        ----------
        i : str/int
            Architecture id.
        """
        
        return self.meta["ids"][str(i)]["isomorph"]
    
    def get_result(self,dataset,architecture_id,metric):
        architecture_id = self.get_uid(architecture_id)
        return self.res[dataset][architecture_id][metric]['1']