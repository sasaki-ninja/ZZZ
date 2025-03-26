from typing import Union, Any, AsyncGenerator, Callable
import asyncio
import bittensor as bt

class EagerDendrite(bt.Dendrite):
    """
    Custom dendrite to get the response of first miner that responds,
    where its reponse passes the filter.

    Note this dendrite has to be async for this to make any sense,
    And that it does not support streaming.
    """

    async def forward(
        self,
        axons: Union[list[Union["bt.AxonInfo", "bt.Axon"]], Union["bt.AxonInfo", "bt.Axon"]],
        synapse: "bt.Synapse",
        filter: Callable[[Union["bt.Synapse", Any]], bool],
        timeout: float = 12,
        deserialize: bool = True,
    ) -> Union["AsyncGenerator[Any, Any]", "bt.Synapse"]:
        """
        See parent class bt.Dendrite for original documentation. 
        Does not support streaming behaviour, since this dendrite is already greedy.
        Is necessarily asynchronous.

        If deserialize is True, the filter is passed the output of the deserialised synapse.
        If deserialise is False, the filter is passed the synapse itself.

        Returns: A single Synapse, the first one that responded and passes filter
        """

        async def query_axons_eager(
            filter: Callable[["bt.Synapse"], bool],
        ) -> Union["AsyncGenerator[Any, Any]", "bt.Synapse"]:
            
            async def single_axon_response(
                target_axon: Union["bt.AxonInfo", "bt.Axon"],
            ) -> Union["AsyncGenerator[Any, Any]", "bt.Synapse"]:
                return await self.call(
                    target_axon=target_axon,
                    synapse=synapse.model_copy(),  # type: ignore
                    timeout=timeout,
                    deserialize=deserialize,
                )

            for task in asyncio.as_completed(
                [single_axon_response(target_axon) for target_axon in axons]
            ):
                result = await task
                if filter(result):
                    return result

        # Get responses eagerily, so return first one that succeeds
        response = await query_axons_eager(filter)
        return response