/* Exact two-state dynamic programming for HV/VH connectors in a fixed
   ordered open path. This is candidate generation, not a GLNS/GLKH solver. */
#include <stdlib.h>
#include <stdint.h>

#define API __declspec(dllexport)

static void arc(const double *a, const double *b, int option,
                int *first, int *last, int *bend) {
    double dx = b[0] - a[0], dy = b[1] - a[1];
    int h = dx > 0 ? 0 : 2, v = dy > 0 ? 1 : 3;
    if (dx == 0) {
        *first = *last = v;
        *bend = 0;
    } else if (dy == 0) {
        *first = *last = h;
        *bend = 0;
    } else {
        *first = option ? v : h;
        *last = option ? h : v;
        *bend = 1;
    }
}

API int open_dp(const double *p, int n, int start_option, int end_option,
                uint8_t *choice, int *first, int *last) {
    if (n < 2)
        return -1;

    int k, s, t, f[2], l[2], bend[2], prev_last[2], old[2], now[2], INF = 1000000000;
    uint8_t *back = (uint8_t *)malloc((size_t)(n - 1) * 2);
    if (!back)
        return -2;

    for (s = 0; s < 2; s++) {
        arc(p, p + 2, s, &f[s], &prev_last[s], &bend[s]);
        old[s] = (start_option < 0 || s == start_option) ? bend[s] : INF;
    }

    for (k = 1; k < n - 1; k++) {
        for (s = 0; s < 2; s++) {
            arc(p + 2 * k, p + 2 * (k + 1), s, &f[s], &l[s], &bend[s]);
            int a = old[0] + (prev_last[0] != f[s]), b = old[1] + (prev_last[1] != f[s]);
            t = a <= b ? 0 : 1;
            now[s] = (t == 0 ? a : b) + bend[s];
            back[2 * k + s] = (uint8_t)t;
        }
        old[0] = now[0];
        old[1] = now[1];
        prev_last[0] = l[0];
        prev_last[1] = l[1];
    }

    s = end_option >= 0 ? end_option : (old[0] <= old[1] ? 0 : 1);
    int result = old[s];
    for (k = n - 2; k >= 0; k--) {
        choice[k] = (uint8_t)s;
        if (k)
            s = back[2 * k + s];
    }

    int dummy, a, b;
    arc(p, p + 2, choice[0], first, &dummy, &a);
    arc(p + 2 * (n - 2), p + 2 * (n - 1), choice[n - 2], &dummy, last, &b);
    free(back);
    return result;
}
