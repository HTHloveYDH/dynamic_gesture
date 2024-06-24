/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/tfrt/common/pjrt_state.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tf_pjrt_client.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_options.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"

namespace tensorflow {

PjRtState* PjRtState::Create() { return new PjRtState(); }

StatusOr<xla::PjRtClient*> PjRtState::GetPjRtClient(
    const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    return it->second.get();
  }
  return errors::NotFound("PjRt client not found for device type ",
                          device_type);
}

StatusOr<xla::PjRtClient*> PjRtState::GetOrCreatePjRtClient(
    const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    return it->second.get();
  }
  std::unique_ptr<xla::PjRtClient> pjrt_client;
  // TODO(b/260799193): use XlaPlatformInfo to pass device-specific options.
  // This info should be set in the plugin init for next pluggable device.

  // TODO(b/280111106): make PjrtClientFactoryOptions an input of
  // GetOrCreatePjRtClient.
  xla::PjrtClientFactoryOptions options = xla::PjrtClientFactoryOptions();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::PjrtClientFactoryRegistry::Get().GetPjrtClient(
                          device_type, options));
  pjrt_client = xla::TfPjRtClient::CreateTfPjRtClient(std::move(client));

  clients_[device_type] = std::move(pjrt_client);
  return clients_[device_type].get();
}

Status PjRtState::SetPjRtClient(const DeviceType& device_type,
                                std::unique_ptr<xla::PjRtClient> client) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    unused_.push_back(std::move(it->second));
  }
  clients_[device_type] = std::move(client);
  return OkStatus();
}

Status PjRtState::MovePjRtClientToUnused(const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    unused_.push_back(std::move(it->second));
    clients_.erase(it);
    return OkStatus();
  }
  return errors::NotFound("PjRt client not found for device type ",
                          device_type);
}

string PjRtState::DebugString() const { return "PjRtState"; }

}  // namespace tensorflow
