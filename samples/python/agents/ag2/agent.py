import asyncio
import logging
import os
import traceback
from dotenv import load_dotenv
from typing import AsyncIterable, Dict, Any

from autogen import AssistantAgent, LLMConfig
from autogen.mcp import create_toolkit

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

def get_api_key() -> str:
    """Helper method to handle API Key."""
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")

class YoutubeMCPAgent:
    """Agent to access a Youtube MCP Server to download closed captions"""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        # Import AG2 dependencies here to isolate requirements
        try:
            # Set up LLM configuration
            llm_config = LLMConfig(
                model="gemini-2.0-flash",
                api_type="google",
                api_key=get_api_key()
            )

            # Create the assistant agent that will use MCP tools
            self.agent = AssistantAgent(
                name="YoutubeMCPAgent",
                llm_config=llm_config,
                system_message="You are a helpful assistant with access to MCP tools. You can solve various tasks using these tools.",
            )

            self.initialized = True
            logger.info("MCP Agent initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import AG2 components: {e}")
            self.initialized = False

    async def _create_toolkit_and_run(self, query: str) -> str:
        """Create MCP toolkit and run the query with the agent."""
        try:
            # Create stdio server parameters for MCP
            server_params = StdioServerParameters(
                command="mcp-youtube",
            )

            logger.info(f"Creating MCP toolkit for query: {query[:50]}...")

            # Connect to the MCP server using stdio client
            async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Create a toolkit with available MCP tools
                toolkit = await create_toolkit(session=session)

                # Register MCP tools with the agent
                toolkit.register_for_llm(self.agent)

                # Make a request using the MCP tools
                result = await self.agent.a_run(
                    message=query,
                    tools=toolkit.tools,
                    max_turns=2,
                    user_input=False,
                )

                try:    
                    # Process the result which will print the output
                    await result.process()

                    # Get the summary which by default contains the agent's last message output
                    return await result.summary

                except Exception as extraction_error:
                    logger.error(f"Error extracting response: {extraction_error}")
                    return f"Error extracting response from MCP agent: {str(extraction_error)}"
        except Exception as e:
            logger.error(f"Error using MCP toolkit: {e}")
            raise

    async def stream(self, query: str, sessionId: str) -> AsyncIterable[Dict[str, Any]]:
        """Stream updates from the MCP agent."""
        if not self.initialized:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "MCP agent initialization failed. Please check the dependencies and logs."
            }
            return

        try:
            # Initial status update
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "Connecting to MCP server..."
            }

            logger.info(f"Processing query in stream: {query[:50]}...")

            try:                
                # Create stdio server parameters for mcp-youtube
                server_params = StdioServerParameters(
                    command="mcp-youtube",
                )

                # Progress update
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Setting up MCP toolkit..."
                }

                # Connect to the MCP server using stdio client
                async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
                    # Initialize the connection
                    await session.initialize()

                    # Progress update
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Processing with MCP tools..."
                    }

                    # Create toolkit and register tools
                    toolkit = await create_toolkit(session=session)
                    toolkit.register_for_llm(self.agent)

                    # Process the request
                    result = await self.agent.a_run(
                        message=query,
                        tools=toolkit.tools,
                        max_turns=2,
                        user_input=False,
                    )

                    # Extract the content from the result
                    try:
                        # Process the result which will print the output
                        await result.process()

                        # Get the summary which contains the output
                        response = await result.summary

                    except Exception as extraction_error:
                        logger.error(f"Error extracting response: {extraction_error}")
                        response = f"Error extracting response from MCP agent: {str(extraction_error)}"

                    # Final response
                    yield {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": response
                    }
            except Exception as e:
                logger.error(f"Error during MCP processing: {traceback.format_exc()}")
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": f"Error during MCP interaction: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error in streaming MCP agent: {traceback.format_exc()}")
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"Error during MCP interaction: {str(e)}"
            }

    def invoke(self, query: str, sessionId: str) -> Dict[str, Any]:
        """Synchronous invocation of the MCP agent."""
        raise NotImplementedError(
            "Synchronous invocation is not supported by this agent. Use the streaming endpoint (tasks/sendSubscribe) instead."
        )
