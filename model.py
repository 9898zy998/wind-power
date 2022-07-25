import paddle
import paddle.nn as nn
class BaselineGruModel(nn.Layer):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out_dim = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out_dim)

    def forward(self, x_enc):
        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])
        x_enc = paddle.concat((x_enc, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out_dim:]
        return sample 
