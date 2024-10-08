from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def stream_logs(container):
    buffer = ''
    try:
        for log_chunk in container.logs(stream=True, follow=True):
            log_text = log_chunk.decode('utf-8', errors='replace')
            buffer += log_text
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line:
                    logger.info(line)
        if buffer:
            logger.info(buffer)
    except Exception as e:
        logger.error(f"Error streaming logs: {str(e)}")