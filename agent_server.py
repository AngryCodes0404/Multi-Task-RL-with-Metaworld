#!/usr/bin/env python3
"""
Cap'n Proto RPC Server for Agent Interface.

Receives observations as structured Observation messages containing Tensor entries
so agents can work with both state vectors and rich sensory data without pickle.
"""

import asyncio
import logging
import os

import capnp
import numpy as np
import torch

# Load the schema
schema_file = os.path.join(os.path.dirname(__file__), "agent.capnp")
agent_capnp = capnp.load(schema_file)

logger = logging.getLogger(__name__)

# Default network configuration
DEFAULT_RPC_ADDRESS = "127.0.0.1"
DEFAULT_RPC_PORT = 8000

_TRAVERSAL_WORDS = 100 * 1024 * 1024  # match client; tune appropriately


class AgentServer(agent_capnp.Agent.Server):
    """Cap'n Proto server implementation for AgentInterface"""

    def __init__(self, agent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.logger.info("AgentServer initialized with agent: %s", type(agent).__name__)

    async def act(self, obs, **kwargs):
        """Handle act RPC call. 'obs' is an Observation struct containing Tensor entries."""
        try:
            entries = list(obs.entries)
            self.logger.debug(
                "Server.act invoked; incoming observation entries=%d",
                len(entries),
            )

            def tensor_to_numpy(tensor_msg) -> np.ndarray:
                array = np.frombuffer(
                    tensor_msg.data, dtype=np.dtype(tensor_msg.dtype)
                ).reshape(tuple(tensor_msg.shape))
                return array.copy()

            if len(entries) == 1 and entries[0].key == "__value__":
                obs_payload = tensor_to_numpy(entries[0].tensor)
            else:
                obs_payload = {
                    entry.key: tensor_to_numpy(entry.tensor) for entry in entries
                }

            # call the underlying agent synchronously
            action_tensor = self.agent.act(obs_payload)

            # convert to numpy
            if isinstance(action_tensor, torch.Tensor):
                action_np = action_tensor.detach().cpu().numpy()
            else:
                action_np = np.array(action_tensor)

            # Build response Tensor
            response = agent_capnp.Tensor.new_message()
            response.data = action_np.tobytes()
            response.shape = [int(s) for s in action_np.shape]
            response.dtype = str(action_np.dtype)
            return response
        except Exception:
            self.logger.exception("Exception in AgentServer.act")
            raise

    async def reset(self, **kwargs):
        try:
            self.agent.reset()
        except Exception:
            self.logger.exception("Error in reset")
            raise

    async def ping(self, message, **kwargs):
        self.logger.info(f"Ping received: {message}")
        return "pong"


async def serve(agent, address=DEFAULT_RPC_ADDRESS, port=DEFAULT_RPC_PORT):
    """Serve the agent using asyncio approach"""

    async def new_connection(stream):
        try:
            server = capnp.TwoPartyServer(
                stream,
                bootstrap=AgentServer(agent),
                traversal_limit_in_words=_TRAVERSAL_WORDS,
            )
            await server.on_disconnect()
        except Exception:
            logger.exception("Error handling connection")

    server = await capnp.AsyncIoStream.create_server(new_connection, address, port)
    logger.info("Agent RPC server listening on %s:%d", address, port)

    try:
        async with server:
            await server.serve_forever()
    except Exception:
        logger.exception("Server error")
    finally:
        logger.info("Server shutting down")


def start_server(agent, address=DEFAULT_RPC_ADDRESS, port=DEFAULT_RPC_PORT):
    async def run_server_with_kj():
        async with capnp.kj_loop():
            await serve(agent, address, port)

    try:
        asyncio.run(run_server_with_kj())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


def run_server_in_process(agent, address=DEFAULT_RPC_ADDRESS, port=DEFAULT_RPC_PORT):
    async def run_with_kj():
        async with capnp.kj_loop():
            await serve(agent, address, port)

    asyncio.run(run_with_kj())
