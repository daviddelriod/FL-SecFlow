import torch

from .single_step import UserSingleStep

class UserMultiStep(UserSingleStep):
    """A user who computes multiple local update steps as in a FedAVG scenario."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, idx, cfg_user)

        self.num_local_updates = cfg_user['num_local_updates']
        self.num_data_per_local_update_step = cfg_user['num_data_per_local_update_step']
        self.local_learning_rate = cfg_user['local_learning_rate']
        self.provide_local_hyperparams = cfg_user['provide_local_hyperparams']

    def __repr__(self):
        n = "\n"
        return (
            super().__repr__()
            + n
            + f"""    Local FL Setup:
        Number of local update steps: {self.num_local_updates}
        Data per local update step: {self.num_data_per_local_update_step}
        Local learning rate: {self.local_learning_rate}

        Threat model:
        Share these hyperparams to server: {self.provide_local_hyperparams}

        """
        )

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""
        self.counted_queries += 1
        user_data = self._load_data()

        # Compute local updates
        server_payload = server_payload[0]
        parameters = server_payload["parameters"]
        buffers = server_payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                self.model.train()
        log.info(
            f"Computing user update on user {self.user_idx} in model mode: {'training' if self.model.training else 'eval'}."
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
        seen_data_idx = 0
        label_list = []
        for step in range(self.num_local_updates):
            data = {
                k: v[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step] for k, v in user_data.items()
            }
            seen_data_idx += self.num_data_per_local_update_step
            seen_data_idx = seen_data_idx % self.num_data_points
            label_list.append(data["labels"].sort()[0])

            optimizer.zero_grad()
            # Compute the forward pass
            data[self.data_key] = (
                data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
                if self.generator_input is not None
                else data[self.data_key]
            )
            #print(data, type(data))
            outputs = self.model(data['inputs'])
            loss = self.loss(outputs, data["labels"])
            loss.backward()

            grads_ref = [p.grad for p in self.model.parameters()]
            if self.clip_value > 0:
                self._clip_list_of_grad_(grads_ref)
            self._apply_differential_noise(grads_ref)
            optimizer.step()

        # Share differential to server version:
        # This is equivalent to sending the new stuff and letting the server do it, but in line
        # with the gradients sent in UserSingleStep
        shared_grads = [
            (p_local - p_server.to(**self.setup)).clone().detach()
            for (p_local, p_server) in zip(self.model.parameters(), parameters)
        ]

        shared_buffers = [b.clone().detach() for b in self.model.buffers()]
        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=user_data["labels"] if self.provide_labels else None,
            local_hyperparams=dict(
                lr=self.local_learning_rate,
                steps=self.num_local_updates,
                data_per_step=self.num_data_per_local_update_step,
                labels=label_list,
            )
            if self.provide_local_hyperparams
            else None,
            data_key=self.data_key,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=user_data[self.data_key], labels=user_data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data
