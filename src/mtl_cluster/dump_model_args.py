from rich.console import Console

from mtl_cluster.clustering import determine_run_id, get_args_from_run_id


console = Console()

if __name__ == "__main__":
    datasets = (
        "GWSD",
        "MBIC",
    )
    models = [
        "cross_attention_unpooled",
        "cross_attention_pooled",
        "encoder_encoder",
        "classifier",
        "decoder_only",
        "encoder_decoder_pretrained",
    ]

    console.rule()
    for model in models:
        for dataset in datasets:
            run_id = determine_run_id(dataset=dataset, model=model)
            console.print(get_args_from_run_id(run_id))
            console.rule()
