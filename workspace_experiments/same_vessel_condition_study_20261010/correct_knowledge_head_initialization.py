"""One bounded correction of the demonstrated auxiliary-readout underfitting."""

from execute_knowledge_trial import main, ROOT


if __name__ == '__main__':
    main(ROOT/'head_initialized', warmstart=True)
