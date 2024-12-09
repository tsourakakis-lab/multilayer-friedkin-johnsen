# The Multilayer Friedkin-Johnsen Model

Understanding the impact of social influence through data analysis is crucial for validating and refining existing models of opinion dynamics. The pivotal research by Friedkin and Bullo demonstrated that the Friedkin-Johnsen (FJ) model aligns closely with real-world opinion patterns, as evidenced by extensive experiments involving actual participants. However, the applicability of this model to networks with multiple layers, such as social media platforms like Twitter and knowledge graphs remains an open question. In this study, we introduce two multilayered enhancements of the FJ model. Our first model allows for variable weighting of interactions across different layers, thus enhancing the versatility of the traditional FJ model. Our second model is a personalized version of the first approach, in which users enjoy their own weighting of importance for each layer. To show the power of the Multilayer FJ (MFJ) model, we prove that its equilibrium can be polarized even when the single FJ model converges to uniform opinions. Also, we experimentally tested the expressiveness of our MFJ model in the task of opinion learning and forecasting, using both synthetic and real-world networks. The findings suggest that our modified MFJ model not only outperforms the standard version but also yields valuable insights into node characteristics and the importance of layers in shaping opinions.

## Synthetic datasets

MFJ_synthetic.ipynb

## Real datasets

MFJ_download_and_make_real_graphs.ipynb

MFJ_real.ipynb

## Scalability Experiments

MFJ_scalability.ipynb
