"""Trace batch-size roundoff versus incorrect native/cache input correspondence."""
import run as r


def main():
    r.runtime('cuda')
    torch = r.torch
    rows, cache = r.cohorts(), r.Cache('cuda')
    net = r.fresh('full', 'cuda')
    queries = r.sample_rows(rows)
    bank = cache.bank(rows['references'])
    diagnostics = []
    with torch.no_grad():
        for record in queries:
            idx = cache.lookup[record['id']]
            start = idx//8*8
            original_group = cache.records[start:start+8]
            native = r.data.batch([r.data.prepare_record(v) for v in original_group], 'cuda')
            same_batch = net.prefix(native['images'], stop=2)
            exact = cache.batch(original_group)
            error = float((same_batch-exact['prefix']).abs().max())
            assert error < 1e-5, ('original_cache_batch', record['id'], error)
            one = r.data.batch([r.data.prepare_record(record)], 'cuda')
            prefix = net.prefix(one['images'], stop=2)
            cached = cache.batch([record])
            fields = {key:float((one[key]-cached[key]).abs().max()) for key in one if key!='images'}
            assert max(fields.values()) == 0
            legal = r.data.pair_mask([record],rows['references'],'cuda')
            native_score = net.forward_from(prefix, **{k:v for k,v in one.items() if k!='images'}, reference=bank,pair_valid=legal)
            cached_score = cache.score(net,[record],rows['references'],bank)
            diagnostics.append(dict(id=record['id'], original_batch_replay=error,
                one_vs_eight_prefix=float((prefix-cached['prefix']).abs().max()),
                prefix_scale=float(cached['prefix'].abs().max()), fields=fields,
                logit_error=float((native_score-cached_score).abs().max()),
                prediction_unchanged=bool(native_score.argmax(1).eq(cached_score.argmax(1)).all())))
        output=dict(rows=diagnostics)
        output['whole_native_logit_error']=r.native_error(net,cache,queries,rows['references'],bank)
        r.write(r.HERE/'cache_roundoff_diagnosis.json',output)
        print(output)


if __name__=='__main__':
    main()
