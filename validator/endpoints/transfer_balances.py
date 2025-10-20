"""
Transfer balance endpoints for coldkey balance information
"""

from fastapi import APIRouter, Depends, HTTPException

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.transfer_models import ColdkeyBalanceResponse
from validator.db.sql.transfers import get_coldkey_balance_by_address
from validator.utils.logging import get_logger

logger = get_logger(__name__)

# Constants for RAO to TAO conversion
RAO_TO_TAO_DIVISOR = 1_000_000_000  # 1 TAO = 1,000,000,000 RAO


def rao_to_tao(rao_amount: int) -> float:
    """
    Convert RAO amount to TAO (for display purposes)
    
    Args:
        rao_amount: Amount in RAO as integer
        
    Returns:
        float: Amount in TAO
    """
    return rao_amount / RAO_TO_TAO_DIVISOR


async def get_coldkey_balance(
    coldkey: str,
    config: Config = Depends(get_config)
) -> ColdkeyBalanceResponse:
    """
    Get balance information for a specific coldkey address
    
    Args:
        coldkey: Coldkey SS58 address
        config: Validator configuration
        
    Returns:
        ColdkeyBalanceResponse: Balance information for the coldkey
        
    Raises:
        HTTPException: If coldkey not found
    """
    logger.info(f"Getting balance for coldkey: {coldkey}")
    
    # Get balance from database
    balance = await get_coldkey_balance_by_address(config.psql_db, coldkey)
    
    if not balance:
        raise HTTPException(
            status_code=404, 
            detail=f"Coldkey balance not found for address: {coldkey}"
        )
    
    # Convert to response format with TAO values
    response = ColdkeyBalanceResponse(
        coldkey=balance.coldkey,
        balance_rao=balance.balance_rao,
        balance_tao=rao_to_tao(balance.balance_rao),
        total_sent_rao=balance.total_sent_rao,
        total_sent_tao=rao_to_tao(balance.total_sent_rao),
        transfer_count=balance.transfer_count,
        last_transfer_at=balance.last_transfer_at,
        created_at=balance.created_at,
        updated_at=balance.updated_at,
    )
    
    logger.info(f"Returning balance for {coldkey}: {balance.balance_rao} RAO ({response.balance_tao:.6f} TAO)")
    return response


def factory_router() -> APIRouter:
    """Factory function to create the transfer balances router"""
    router = APIRouter()
    
    router.add_api_route(
        "/transfer/balance/{coldkey}",
        get_coldkey_balance,
        methods=["GET"],
        response_model=ColdkeyBalanceResponse,
        summary="Get coldkey balance information",
        description="Get balance and transfer information for a specific coldkey address",
        tags=["transfers"]
    )
    
    return router




