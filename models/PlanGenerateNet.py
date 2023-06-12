import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(75, 256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 36))
    net.apply(init_weights)

    return net


class LSTMTaggerSep(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, lstm_num_layers):
        super(LSTMTaggerSep, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 1
        self.human_input_embedding = nn.Linear(159, self.embedding_dim)
        self.batch_size = 1

        self.num_layers = lstm_num_layers
        #self.robot_input_embedding = nn.Linear(84, self.hidden_dim)
        self.lstm_c_embedding = nn.Linear(84, self.hidden_dim)

        self.lstm_xyz = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=lstm_num_layers)
        self.lstm_vel = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=lstm_num_layers)
        self.lstm_for = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=lstm_num_layers)

        self.output_xyz_embedding = nn.Linear(self.hidden_dim, 36)
        self.output_vel_embedding = nn.Linear(self.hidden_dim, 36)
        self.output_for_embedding = nn.Linear(self.hidden_dim, 12)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
    def forward(self, human_inputs):

        human_embeddings = self.human_input_embedding(human_inputs)

        h_0_xyz = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_xyz = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        h_0_vel = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_vel = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        h_0_for = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_for = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        lstm_xyz_out, _ = self.lstm_xyz(human_embeddings, (h_0_xyz, c_0_xyz))
        lstm_vel_out, _ = self.lstm_vel(human_embeddings, (h_0_vel, c_0_vel))

        output_embedding_xyz = self.leaky_relu(self.output_xyz_embedding(lstm_xyz_out))
        output_embedding_vel = self.tanh(self.leaky_relu(self.output_vel_embedding(lstm_vel_out)))

        return output_embedding_xyz, output_embedding_vel  #, output_embedding_for
    


class LSTMPreSep(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, lstm_num_layers):
        super(LSTMPreSep, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 1
        self.human_input_embedding = nn.Linear(159, self.embedding_dim)
        self.batch_size = 1

        self.num_layers = lstm_num_layers
        #self.robot_input_embedding = nn.Linear(84, self.hidden_dim)
        self.lstm_c_embedding = nn.Linear(84, self.hidden_dim)

        self.lstm_xyz = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=lstm_num_layers)
        self.lstm_vel = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=lstm_num_layers)
        self.lstm_for = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=lstm_num_layers)

        self.output_xyz_embedding = nn.Linear(self.hidden_dim, 36)
        self.output_vel_embedding = nn.Linear(self.hidden_dim, 36)
        self.output_for_embedding = nn.Linear(self.hidden_dim, 12)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, human_inputs):

        human_embeddings = self.human_input_embedding(human_inputs)

        h_0_xyz = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_xyz = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        h_0_vel = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_vel = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        h_0_for = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_for = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        lstm_xyz_out, _ = self.lstm_xyz(human_embeddings, (h_0_xyz, c_0_xyz))
        lstm_vel_out, _ = self.lstm_vel(human_embeddings, (h_0_vel, c_0_vel))
        #lstm_vel_for, _ = self.lstm_for(human_embeddings, (h_0_for, c_0_for))

        output_embedding_xyz = self.leaky_relu(self.output_xyz_embedding(lstm_xyz_out))
        output_embedding_vel = self.tanh(self.leaky_relu(self.output_vel_embedding(lstm_vel_out)))
        #output_embedding_for = self.leaky_relu(self.output_for_embedding(lstm_vel_out))

        return output_embedding_xyz, output_embedding_vel #, output_embedding_for



class TransformerTraggerSeq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(TransformerTraggerSeq, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 1
        self.human_input_embedding = nn.Linear(159, self.embedding_dim)
        self.batch_size = 1

        self.num_layers = num_layers
        #self.robot_input_embedding = nn.Linear(84, self.hidden_dim)
        self.trans_c_embedding = nn.Linear(84, self.hidden_dim)

        self.trans_xyz = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_layers,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=self.hidden_dim
        )
        self.trans_vel = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_layers,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=self.hidden_dim
        )
        self.trans_for = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_layers,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=self.hidden_dim
        )

        self.output_xyz_embedding = nn.Linear(self.embedding_dim, 36)
        self.output_vel_embedding = nn.Linear(self.embedding_dim, 36)
        self.output_for_embedding = nn.Linear(self.embedding_dim, 12)

        self.relu = nn.ReLU()

    def forward(self, human_inputs):
        human_embeddings = self.human_input_embedding(human_inputs)

        h_0_xyz = torch.ones(1, 1, self.embedding_dim).to("mps")
        h_0_vel = torch.ones(1, 1, self.embedding_dim).to("mps")

        trans_xyz_out = self.trans_xyz(human_embeddings, h_0_xyz)
        trans_vel_out = self.trans_vel(human_embeddings, h_0_vel)

        output_embedding_xyz = self.relu(self.output_xyz_embedding(trans_xyz_out))
        output_embedding_vel =self.relu(self.output_vel_embedding(trans_vel_out))

        return output_embedding_xyz, output_embedding_vel
    


class KimoreFusionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(KimoreFusionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 1
        self.human_input_embedding = nn.Linear(159, self.embedding_dim)
        self.batch_size = 1

        self.num_layers = num_layers
        #self.robot_input_embedding = nn.Linear(84, self.hidden_dim)
        self.lstm_c_embedding = nn.Linear(84, self.hidden_dim)

        self.lstm_xyz = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers)

        # with a great performance 
        self.trans_vel = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_layers,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=self.hidden_dim
        )

        self.trans_for = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_layers,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=self.hidden_dim
        )

        self.output_xyz_embedding = nn.Linear(self.hidden_dim, 36)
        self.output_vel_embedding = nn.Linear(self.embedding_dim, 36)
        self.output_for_embedding = nn.Linear(self.embedding_dim, 12)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, human_inputs):
        human_embeddings = self.human_input_embedding(human_inputs)

        h_0_xyz = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")
        c_0_xyz = torch.ones(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to("mps")

        h_0_vel = torch.ones(1, 1, self.embedding_dim).to("mps")

        xyz_out, _ = self.lstm_xyz(human_embeddings, (h_0_xyz, c_0_xyz))
        vel_out = self.trans_vel(human_embeddings, h_0_vel)

        output_embedding_xyz = self.tanh(self.leaky_relu(self.output_xyz_embedding(xyz_out)))
        output_embedding_vel =self.relu(self.output_vel_embedding(vel_out))

        return output_embedding_xyz, output_embedding_vel
        