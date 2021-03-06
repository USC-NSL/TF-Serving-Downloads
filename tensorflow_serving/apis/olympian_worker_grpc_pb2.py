# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from tensorflow_serving.apis import predict_pb2 as tensorflow__serving_dot_apis_dot_predict__pb2


class OlympianWorkerStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/tensorflow.serving.OlympianWorker/Predict',
        request_serializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictRequest.SerializeToString,
        response_deserializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.FromString,
        )


class OlympianWorkerServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Predict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_OlympianWorkerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictRequest.FromString,
          response_serializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.serving.OlympianWorker', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
