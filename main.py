import logging
from training import Training


if __name__ == "__main__":
    # set logger here
    logging.basicConfig(filename="test.log",
                        level=logging.DEBUG)
    # try:
    trainer = Training("parameters.json")
    logging.info("creater training object...")
    trainer.train()
    # if success:
    logging.info("Training done")
    # except:
    #     logging.critical("something went very wrongggggg!!!!!!!!")

