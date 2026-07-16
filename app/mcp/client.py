"""MCP (Model Context Protocol) client for connecting to and interacting with MCP servers.

Provides a reusable async client that can connect to Python or Node.js MCP servers
via stdio transport, list available tools, and execute tool calls.

Usage::

    client = MCPClient()
    try:
        await client.connect_to_server("app/mcp/amap_server.py")
        result = await client.execute("maps_weather", {"city": "Beijing"})
    finally:
        await client.cleanup()
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.shared.logging import get_logger

logger = get_logger(__name__)


class MCPClient:
    """Async MCP client that connects to a server over stdio and invokes tools."""

    def __init__(self) -> None:
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._stdio: Any = None
        self._write: Any = None

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to an MCP server script and list its available tools.

        Args:
            server_script_path: Path to a ``.py`` or ``.js`` MCP server script.

        Raises:
            ValueError: If the script path does not end with ``.py`` or ``.js``.
        """
        if server_script_path.endswith(".py"):
            command = "python"
        elif server_script_path.endswith(".js"):
            command = "node"
        else:
            raise ValueError("Server script must be a .py or .js file")

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._stdio, self._write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self._stdio, self._write)
        )

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        logger.info(
            "Connected to MCP server %s, available tools: %s",
            server_script_path,
            [tool.name for tool in tools],
        )

    async def execute(self, function_name: str, tool_args: Dict[str, Any]) -> str:
        """Call a tool on the connected MCP server.

        Args:
            function_name: Name of the MCP tool to invoke.
            tool_args: Keyword arguments to pass to the tool.

        Returns:
            The text content of the first result item, or ``"Not Found"`` on error.
        """
        logger.info("Executing tool %s with args %s", function_name, tool_args)

        try:
            result = await self.session.call_tool(function_name, tool_args)
            text = result.content[0].text
            logger.info("MCP response for %s: %s", function_name, text)
            return text
        except Exception:
            logger.exception("Error executing MCP tool %s", function_name)
            return "Not Found"

    async def cleanup(self) -> None:
        """Release all resources held by the client."""
        await self.exit_stack.aclose()


async def main() -> None:
    """Quick smoke-test of the MCP client against the Amap server."""
    client = MCPClient()
    try:
        await client.connect_to_server("app/mcp/amap_server.py")
        await client.execute("maps_weather", {"city": "北京"})
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
