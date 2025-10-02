from soundscapy.sspylogging import get_logger

from ._r_wrapper import get_r_session, install_r_packages

logger = get_logger()

install_r_packages()

_, _, _, _, _, rthorr = get_r_session()
logger.debug("R session and packages retrieved successfully.")
