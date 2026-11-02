/* rdec cpp corpus: namespace and nested namespace scopes (?f@N@@, ?g@inner@outer@@) */
namespace geo { namespace detail { int clamp(int v, int lo, int hi); } int norm(int v); }
int geo::detail::clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
int geo::norm(int v) { return detail::clamp(v, 0, 10); }
