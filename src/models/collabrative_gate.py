import torch
import torch.nn as nn
import torch.nn.functional as F


class Speaker_Independent_Single_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=2):
        super(Speaker_Independent_Single_Mode_without_Context, self).__init__()
        print("No. of classes:",num_classes)
        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            # nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            # nn.BatchNorm1d(2*self.shared_embedding),
            # nn.ReLU(),
            # nn.Linear(2*self.shared_embedding, self.shared_embedding),
            # nn.BatchNorm1d(self.shared_embedding),
            # nn.ReLU(),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calcuates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_utterance)

        input = updated_shared

        return self.pred_module(input)


class Speaker_Independent_Dual_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=2):
        super(Speaker_Independent_Dual_Mode_without_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA,  uB):
        """making Feature Projection in order to make all feature of same dimension"""

        # if uA contain text lenth, then we need to mean the text length
        if len(uA.shape) == 3:
            uA = torch.mean(uA, dim=1)
        if len(uB.shape) == 3:
            uB = torch.mean(uB, dim=1)
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,  shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,  shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        return self.pred_module(input)


class Speaker_Independent_Triple_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=1024, shared_embedding=1024, projection_embedding=512, dropout=0.3, num_classes=5):
        super(Speaker_Independent_Triple_Mode_without_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calcuates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + self.attention(feA, feC)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, uB,  uC):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,   shared_C_utterance,  shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance,   shared_A_utterance,  shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,   shared_A_utterance,  shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        return self.pred_module(input)