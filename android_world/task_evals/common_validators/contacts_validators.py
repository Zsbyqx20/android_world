# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logic for validating contact has been added."""

from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.utils import user_data_generation
from android_world.utils import contacts_utils


class AddContact(task_eval.TaskEval):
  """Validator for checking that a contact has been added."""

  app_names = ()
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {
          'name': {'type': 'string'},
          'number': {'type': 'string'},
      },
      'required': ['name', 'number'],
  }
  template = ''
  noise_num = 0

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    contacts_utils.clear_contacts(env.controller)
    if self.noise_num > 0:
      # Create target contact for comparison
      target_contact = contacts_utils.Contact(
          self.params['name'],
          contacts_utils.clean_phone_number(self.params['number']),
      )

      # Keep track of generated noise contacts to avoid duplicates
      generated_contacts = set()

      for _ in range(self.noise_num):
        # Generate noise contact that doesn't match target
        while True:
          noise_name = user_data_generation.generate_random_name()
          noise_number = user_data_generation.generate_random_number()
          noise_contact = contacts_utils.Contact(
              noise_name,
              contacts_utils.clean_phone_number(noise_number),
          )

          # Ensure noise contact doesn't match target and isn't duplicate
          if (noise_contact != target_contact and
              noise_contact not in generated_contacts):
            generated_contacts.add(noise_contact)
            contacts_utils.add_contact(noise_name, noise_number, env.controller)
            break

  def _has_contact(self, contacts: list[contacts_utils.Contact]) -> bool:
    return (
        contacts_utils.Contact(
            self.params['name'],
            contacts_utils.clean_phone_number(self.params['number']),
        )
        in contacts
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    contact_found = self._has_contact(
        contacts_utils.list_contacts(env.controller)
    )
    return super().is_successful(env) if contact_found else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        'name': user_data_generation.generate_random_name(),
        'number': user_data_generation.generate_random_number(),
    }
