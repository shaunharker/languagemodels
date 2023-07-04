# languagemodels


### unused but potentially future useful code

```python
class LinearAutoregression(Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = Linear(d_model, d_model, bias=False)
        self.linear.weight.data *= 0.0

    def forward(self, x):
        xs = []
        n_ctx = x.shape[-1]
        for idx in range(n_ctx):
            if idx == 0:
                xs.append(x[...,idx,:])
            else:
                xs.append(x[...,idx,:] + self.linear(xs[-1]))
        return torch.stack(xs,dim=-2)
```

```python
    def empirical_training(self, xy, q, k):
        """
        Computes the combined loss function using the ground truth labels and an empirical probability distribution.

        Args:
            xy (torch.Tensor): A tensor of shape (batch_size, seq_len + 1) representing input-output pairs of n+1 bytes of text.
            q (torch.Tensor): A tensor of shape (batch_size, seq_len, n_vocab_out) representing the unnormalized empirical probability distribution for predicting y[i] given x[:i+1].
            k (float): A float value between 0.0 and 1.0 representing the balance between training against ground truth (y) and the empirical distribution (q).

        Returns:
            torch.Tensor: A scalar tensor representing the combined loss value.
        """
        (x, y) = self.split_example(xy)
        x = self.module(x)
        y_loss = self.crossentropyloss(x, y)

        # Normalize q along the last dimension
        q = q / torch.sum(q, dim=-1, keepdim=True)

        # Compute predicted probabilities
        p = self.softmax(x)

        # Compute the q_loss: -sum(q_i * log(p_i)) along the last dimension
        q_loss = -torch.sum(q * torch.log(p + 1e-9), dim=-1)

        # Compute the average loss across batch and sequence
        q_loss = q_loss.mean()

        return k * y_loss + (1 - k) * q_loss

        ...
```

```python
    from IPython.display import display, HTML

    async def autocomplete_async(model, encode, decode, prompt=None, n_ctx=None, temp=1.0,
                                n_generate=512, n_vocab_out=None, device=None, verbose=False, output_layer=-1):
        Categorical = torch.distributions.Categorical
        if n_ctx is None:
            n_ctx = model.n_ctx
        if prompt is None:
            prompt = default_prompt
        if device is None:
            device = model.device
        n_vocab_out = n_vocab_out or model.n_vocab_out
        x = encode(prompt)
        x = x[-n_ctx:]
        prompt = decode(x)
        if verbose:
            print(f"=== Prompt ===\n{prompt}\n=== Autocompletion ===\n")

        async def sampler(x):
            x = list(x)
            for _ in range(n_generate):
                await asyncio.sleep(0)  # Give other tasks a chance to run
                #print(model.inference(torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)).shape)
                probs = model.inference(torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0))[output_layer].view(-1)[-n_vocab_out:]
                if temp > 0:
                    y = Categorical(probs=probs**(1.0/temp)).sample().item()
                else:
                    y = torch.argmax(probs).item()
                x = (x + [y])[-n_ctx:]
                yield y

        display_handle = display(HTML(prompt), display_id=True)
        list_of_tokens = []
        async for c in sampler(x):
            list_of_tokens.append(c)
            completion = decode(list_of_tokens)
            def cleanup(s):
                return s.replace('<', '&lt;').replace('>', '&gt;')
            contents = ('<pre style="background: black; color: lime; font-family: monospace">'+
                    '<span style="color: white">' + cleanup(prompt) + '</span>' + cleanup(completion) + '\n'*20 + '</pre>')
            display_handle.update(HTML(contents))

```