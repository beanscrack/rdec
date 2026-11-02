/* rdec cpp corpus: virtual base + override across the vbtable (the vcall-offset territory) */
struct VBase { virtual int vf() const; virtual int vg() const; int v; };
struct VMid : virtual VBase { int vf() const override; int via_base() const; int m; };
struct VLeaf : VMid { int vg() const override; int leaf; };
int VBase::vf() const { return v; }
int VBase::vg() const { return v + 1; }
int VMid::vf() const { return v + m; }
int VMid::via_base() const { const VBase *b = this; return b->vg(); }
int VLeaf::vg() const { return leaf; }
