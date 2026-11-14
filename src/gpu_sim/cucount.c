// CUPTI callback-API call counter. Load into any CUDA process with
//   CUDA_INJECTION64_PATH=/path/libcucount.so CUCOUNT_OUT=counts.json <cmd>
// Counts every driver-API call (runtime calls reach the driver, so each logical call counts once)
// between cuProfilerStart and cuProfilerStop, by name, and writes {"name": count} JSON at exit.
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long counts[CUPTI_DRIVER_TRACE_CBID_SIZE];
static int capturing;

static void CUPTIAPI on_call(void *user, CUpti_CallbackDomain dom, CUpti_CallbackId cbid, const void *data) {
    const CUpti_CallbackData *cb = data;
    if (cb->callbackSite != CUPTI_API_ENTER) return;
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuProfilerStart) capturing = 1;
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuProfilerStop) capturing = 0;
    if (capturing) __atomic_add_fetch(&counts[cbid], 1, __ATOMIC_RELAXED);
}

static void dump(void) {
    const char *path = getenv("CUCOUNT_OUT");
    FILE *f = path ? fopen(path, "w") : stderr;
    const char *sep = "", *name;
    fprintf(f, "{");
    for (int i = 0; i < CUPTI_DRIVER_TRACE_CBID_SIZE; i++) {
        if (!counts[i] || cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, i, &name)) continue;
        fprintf(f, "%s\"%s\": %llu", sep, name, counts[i]);
        sep = ", ";
    }
    fprintf(f, "}\n");
    if (path) fclose(f);
}

int InitializeInjection(void) {
    CUpti_SubscriberHandle sub;
    if (cuptiSubscribe(&sub, on_call, NULL) || cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_DRIVER_API)) return 0;
    atexit(dump);
    return 1;
}
