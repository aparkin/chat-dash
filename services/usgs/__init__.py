"""USGS services initialization."""

from .service import USGSWaterService
from .water_quality import WaterQualityService, GeoBounds, SiteLocation
from .direct_api import WaterQualityAPI

def get_service():
    """Get the USGS water service instance."""
    return USGSWaterService()

__all__ = [
    'USGSWaterService',
    'WaterQualityService',
    'GeoBounds',
    'SiteLocation',
    'WaterQualityAPI',
    'get_service'
] 