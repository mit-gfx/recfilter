#include "recfilter_utils.h"

std::ostream &operator<<(std::ostream &s, RecFilter::CheckResult v) {
    Halide::Image<float> ref = v.ref;
    Halide::Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Halide::Image<float> diff(width, height, channels);

    float diff_sum = 0.0f;
    float max_val  = 0.0f;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = float(ref(x,y,z)) - float(out(x,y,z));
                diff_sum += std::abs(diff(x,y,z) * diff(x,y,z));
                max_val = std::max(ref(x,y,z), max_val);
            }
        }
    }
}

std::ostream &operator<<(std::ostream &s, RecFilter::CheckResultVerbose v) {
    Halide::Image<float> ref = v.ref;
    Halide::Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Halide::Image<float> diff(width, height, channels);

    float diff_sum = 0.0f;
    float max_val  = 0.0f;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = float(ref(x,y,z)) - float(out(x,y,z));
                diff_sum += std::abs(diff(x,y,z) * diff(x,y,z));
                max_val = std::max(ref(x,y,z), max_val);
            }
        }
    }
    float mse  = diff_sum/float(width*height*channels);

    s << "Reference" << "\n" << ref << "\n";
    s << "Halide output" << "\n" << out << "\n";
    s << "Difference " << "\n" << diff << "\n";
    s << "Mean sq error = " << mse << "\n\n";

    return s;
}

std::ostream &operator<<(std::ostream &s, Halide::Func f) {
    s << f.function();
    return s;
}

std::ostream &operator<<(std::ostream &s, Halide::Internal::Function f) {
    if (f.has_pure_definition()) {
        std::vector<Halide::Expr> pure_value = f.values();
        s << "Func " << f.name() << ";\n";
        for (int v=0; v<pure_value.size(); v++) {
            std::vector<std::string> args = f.args();
            s << f.name() << "(";
            for (int i=0; i<args.size(); i++) {
                s << args[i];
                if (i<args.size()-1)
                    s << ",";
            }
            if (pure_value.size()>1)
                s << ")[" << v << "]";
            else
                s << ")";
            s << " = " << pure_value[v] << "\n";
        }

        // reduction definitions
        for (int j=0; j<f.reductions().size(); j++) {
            std::vector<Halide::Expr> reduction_value = f.reductions()[j].values;
            for (int v=0; v<reduction_value.size(); v++) {
                std::vector<Halide::Expr> args = f.reductions()[j].args;
                s << f.name() << "(";
                for (int i=0; i<args.size(); i++) {
                    s << args[i];
                    if (i<args.size()-1)
                        s << ",";
                }
                if (reduction_value.size()>1)
                    s << ")[" << v << "]";
                else
                    s << ")";
                s << " = " << reduction_value[v];
                if (f.reductions()[j].domain.defined()) {
                    s << " with  ";
                    for (int k=0; k<f.reductions()[j].domain.domain().size(); k++) {
                        std::string r = f.reductions()[j].domain.domain()[k].var;
                        s << r << "("
                            << f.reductions()[j].domain.domain()[k].min   << ","
                            << f.reductions()[j].domain.domain()[k].extent<< ") ";
                    }
                }
                s << "\n";
            }
        }
    }
    return s;
}

// -----------------------------------------------------------------------------


Timer::Timer(std::string name) {
    m_Name = name;
    start();
}

Timer::~Timer(void) {
    stop();
}

void Timer::start(void) {
    m_TmStart = milliseconds();
}

Timer::t_time Timer::stop(void) {
    Timer::t_time tm = elapsed();
    Timer::t_time h  = ( tm/(1000*60*60));
    Timer::t_time m  = ((tm/(1000*60)) % 60);
    Timer::t_time s  = ((tm/1000)      % 60);
    Timer::t_time ms = tm % 1000;
    std::cerr << m_Name.c_str() << ": " << h << "h " << m
        << "m " << s << "s " << ms << "ms" << std::endl;
    return tm;
}

Timer::t_time Timer::elapsed(void) {
    return (milliseconds() - m_TmStart);
}

Timer::t_time Timer::milliseconds(void) {
    static bool init = false;
#ifdef WIN32
    static Timer::t_time freq;
    if (!init) {
        init = true;
        LARGE_INTEGER lfreq;
        assert(QueryPerformanceFrequency(&lfreq) != 0);
        freq = Timer::t_time(lfreq.QuadPart);
    }
    LARGE_INTEGER tps;
    QueryPerformanceCounter(&tps);
    return (Timer::t_time(tps.QuadPart)*1000/freq);
#else
    struct timeval        now;
    static struct timeval start;
    if (!init) {
        gettimeofday(&start, NULL);
        init = true;
    }
    gettimeofday(&now, NULL);
    uint ms = uint((now.tv_sec-start.tv_sec)*1000+(now.tv_usec-start.tv_usec)/1000);
    return ms;
#endif
}
