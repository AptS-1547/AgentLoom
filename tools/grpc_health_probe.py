"""Minimal gRPC health probe for the local BERT inference backend."""

from __future__ import annotations

import argparse
import sys

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _build_health_messages():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "health_dynamic.proto"
    file_proto.package = "grpc.health.v1"
    file_proto.syntax = "proto3"

    request = file_proto.message_type.add()
    request.name = "HealthCheckRequest"
    service_field = request.field.add()
    service_field.name = "service"
    service_field.number = 1
    service_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    service_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    response = file_proto.message_type.add()
    response.name = "HealthCheckResponse"
    enum = response.enum_type.add()
    enum.name = "ServingStatus"
    for name, number in (
        ("UNKNOWN", 0),
        ("SERVING", 1),
        ("NOT_SERVING", 2),
        ("SERVICE_UNKNOWN", 3),
    ):
        value = enum.value.add()
        value.name = name
        value.number = number

    status_field = response.field.add()
    status_field.name = "status"
    status_field.number = 1
    status_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_ENUM
    status_field.type_name = ".grpc.health.v1.HealthCheckResponse.ServingStatus"
    status_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    service = file_proto.service.add()
    service.name = "Health"
    method = service.method.add()
    method.name = "Check"
    method.input_type = ".grpc.health.v1.HealthCheckRequest"
    method.output_type = ".grpc.health.v1.HealthCheckResponse"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)

    request_cls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("grpc.health.v1.HealthCheckRequest")
    )
    response_cls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("grpc.health.v1.HealthCheckResponse")
    )
    status_enum = pool.FindEnumTypeByName("grpc.health.v1.HealthCheckResponse.ServingStatus")
    return request_cls, response_cls, status_enum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:50051", help="gRPC target")
    parser.add_argument(
        "--service",
        default="bert_inference.BERTInference",
        help="gRPC health service name, empty string checks overall server",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="timeout in seconds")
    args = parser.parse_args()

    request_cls, response_cls, status_enum = _build_health_messages()
    channel = grpc.insecure_channel(args.target)

    try:
        grpc.channel_ready_future(channel).result(timeout=args.timeout)
        rpc = channel.unary_unary(
            "/grpc.health.v1.Health/Check",
            request_serializer=lambda message: message.SerializeToString(),
            response_deserializer=response_cls.FromString,
        )
        response = rpc(request_cls(service=args.service), timeout=args.timeout)
        status_name = status_enum.values_by_number[response.status].name
        service_name = args.service or "<overall>"
        print(f"target={args.target} service={service_name} status={status_name}")
        return 0 if status_name == "SERVING" else 1
    except grpc.RpcError as exc:
        print(
            f"target={args.target} service={args.service or '<overall>'} "
            f"rpc_error code={exc.code()} details={exc.details()}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            f"target={args.target} service={args.service or '<overall>'} "
            f"probe_error {exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
