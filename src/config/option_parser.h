#pragma once

#include "server_options.h"

MultimodalServerOptions ParseMultimodalOptions(int argc, char** argv);
bool IsHelpRequested(int argc, char** argv);
void PrintUsage(const char* program);
