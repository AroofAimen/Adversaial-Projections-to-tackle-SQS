import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist


class ProtoNet(AbstractMetaLearner):
    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        # print(support_images.shape, query_images.shape)
        z_support, z_query = self.extract_features(support_images, query_images)
        # print("self.training", self.training, "self.no_test_query: ", self.no_test_query)

        # If a transportation method in the feature space has been defined, use it
        if self.training and self.use_train_query and self.transportation_module:
            z_support, z_query = self.transportation_module(z_support, z_query)
        if (not self.training) and self.use_test_query and self.transportation_module:
            z_support, z_query = self.transportation_module(z_support, z_query)
            
        z_proto = self.get_prototypes(z_support, support_labels)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
